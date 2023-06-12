import time
import os
import pandas as pd
from datetime import datetime
from progress.bar import Bar as Bar


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import models as torch_models


import models_cifar
import models_cifar_1_1
import models_imagenet
import models_imagenet_1_1

from utils.utils_EWGS import *
from utils.arg_parser import get_parser
from utils.utils import *
from quantization.weight.AutomaticPruneQuantizer import APQConv2d

from utils.utils import DistributionLoss as KD_loss


def train_model(args):
    print(args)
    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    name_dir = args.name + "__" + datestring
    log_dir = os.path.join(args.output_dir, name_dir)
    os.makedirs(log_dir)

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    print(args.gpus)
    print(len(args.gpus))

    if args.act_quantization_mode is not None and args.act_quantization_mode.startswith("binary"):
        model_zoo = models_cifar_1_1.__dict__ if args.dataset == "cifar10" else models_imagenet_1_1.__dict__
    else:
        model_zoo = models_cifar.__dict__ if args.dataset == "cifar10" else models_imagenet.__dict__

    model = model_zoo[args.arch](quantization_mode=args.act_quantization_mode)

    if args.kd:
        print("Using knowledge distillation")
        #Teacher from pytorch model zoo
        args.teacher = "resnet34"
        model_teacher = torch_models.__dict__[args.teacher](pretrained=True)
        model_teacher = nn.DataParallel(model_teacher).cuda()
        for p in model_teacher.parameters():
            p.requires_grad = False
        model_teacher.eval()
        criterion_kd = KD_loss()


    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    train_loader, val_loader = get_dataloaders(args)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    no_decay = ["alpha", "delta", "beta"]
    non_optimized = ["uA", "lA", "output_scale"]

    param_optimizer = list(model.named_parameters())
    if args.weight_decay_apq == -1:
        weight_decay_apq = args.weight_decay
    else:
        weight_decay_apq = args.weight_decay_apq
    if args.lr_apq == -1:
        lr_apq = args.lr
    else:
        lr_apq = args.lr_apq

    initialize_alpha_delta = True

    if args.act_quantization_mode:
        if args.act_quantization_mode.startswith("ewgs"):
            print("Adjusting EWGS parameters optimization")
            ewgs_parameters = [p for n, p in param_optimizer if any(nd in n for nd in non_optimized)]
            for n, p in param_optimizer:
                if any(nd in n for nd in non_optimized):
                    print(n, p)
            if args.optimizer_q == "Adam":
                optimizer_q = torch.optim.Adam(ewgs_parameters, args.lr_q,
                                             weight_decay=0)
            else:
                optimizer_q = torch.optim.SGD(ewgs_parameters, args.lr_q,
                                               weight_decay=0)
            lr_scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=args.epochs)
        #todo questo va spostato da qua, il learning rate di alpha e delta potrebbe cambiare anche se non stiamo binarizzando
        #todo Va passato un optimizer in piu'alle funzioni di train
        elif args.act_quantization_mode.startswith("binary"):
            print(f"Creating a specific optimizer of alpha and beta using lr {args.lr_apq} and weight decay {args.weight_decay_apq}")
            for module in model.modules():
                if isinstance(module, APQConv2d):
                    module.init_alpha_delta()
            #reload the parameters of the model after adding alpha and delta
            param_optimizer = list(model.named_parameters())

            apb_parameters = [p for n, p in param_optimizer if any(nd in n for nd in no_decay)]

            initialize_alpha_delta = False
            for n, p in param_optimizer:
                    if any(nd in n for nd in no_decay):
                        print(n, p)
            if args.optimizer_q == "Adam":
                optimizer_q = torch.optim.Adam(apb_parameters, lr_apq,
                                               weight_decay=weight_decay_apq)
            else:
                optimizer_q = torch.optim.SGD(apb_parameters, lr_apq,
                                              weight_decay=weight_decay_apq)
            lr_scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=args.epochs)
        else:
            #Dorefa
            non_optimized = []
            optimizer_q = None
            lr_scheduler_q = None

    else:
        non_optimized = []
        optimizer_q = None
        lr_scheduler_q = None


    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(nd in n for nd in non_optimized)],
         'weight_decay': args.weight_decay},
        #todo attenzione decommentando questa riga si evita che i due optimizer modifichino contemporaneamente alpha e delta

        # {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not any(nd in n for nd in non_optimized)],
        #  'weight_decay': weight_decay_apq, 'lr':lr_apq}
    ]

    if args.optim == "Adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError("Wrong optimizer name")

    if args.scheduler == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=args.epochs - args.warmup_epochs, eta_min=0)
    elif args.scheduler == "mstep":
        milestones = args.epochs * np.array([0.6, 0.7, 0.9])
        milestones = list(milestones.astype(np.int64))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
    else:
        raise ValueError("Wrong learning rate scheduler name")

    writer = SummaryWriter(log_dir=log_dir)

    columns = ["epoch", "top1", "n_bits"]
    df_log = pd.DataFrame(columns=columns)

    best_prec1 = -1
    best_epoch = -1
    best_bits = 0
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading pretrained model from ", args.resume)
            sd = torch.load(args.resume)
            if "epoch" in sd.keys():
                start_epoch = sd['epoch']
                model.load_state_dict(sd['state_dict'])
                optimizer.load_state_dict(sd['optimizer'])
                lr_scheduler.load_state_dict(sd['lr_scheduler'])
            else:
                #Old checkpoint
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    if initialize_alpha_delta:
        for module in model.modules():
            if isinstance(module, APQConv2d):
                module.init_alpha_delta()

    model = torch.nn.DataParallel(model)
    model.cuda()
    if args.re_init_weight:
        print("Resetting weights")
        for module in model.modules():
            if isinstance(module, APQConv2d):
                module.reset_bin_params()
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
        print("Training with automatic mixed precision")
    else:
        scaler = None
    train_func = train_dali if args.dali else train
    valid_func = validate_dali if args.dali else validate

    for epoch in range(start_epoch, args.epochs):
        if args.warmup_epochs != 0 and epoch <= args.warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch + 1) / (args.warmup_epochs +1)

        if epoch != 0 and epoch % args.update_scales_every == 0 \
                and args.act_quantization_mode \
                and args.act_quantization_mode.startswith("ewgs") \
                and args.act_quantization_mode.split("-")[-1] == "hessian":

            update_grad_scales(model, train_loader, criterion, torch.device("cuda"), args)

        if epoch == args.freezing_epoch_alpha_delta:
            print("Freezing alpha and delta")
            freeze_alpha_delta(model)

        if epoch == args.freezing_epoch_mask:
            print("Freezing mask")
            freeze_mask(model)

        #if args.quantize_activations and args.act_quantization_mode == "ir-net":
        #    update_irnet_params(args.epochs, epoch, model)

        # train for one epoch
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))
        print('Current lr {:.5f}'.format(optimizer.param_groups[0]['lr']))
        if args.kd:
            train_kd(train_loader, model, model_teacher, criterion_kd, optimizer, scaler=scaler, optimizer_q=optimizer_q)
        else:
            train_func(train_loader, model, criterion, optimizer, scaler, optimizer_q=optimizer_q)

        # evaluate on validation set
        prec1 = valid_func(val_loader, model, criterion, scaler=scaler if args.amp_vali else None)

        if args.warmup_epochs == 0 or epoch >= args.warmup_epochs:
            lr_scheduler.step()

        if lr_scheduler_q:
            lr_scheduler_q.step()
        n_bits = print_stats(model, args, epoch)

        for name, module in model.named_modules():
            if isinstance(module, APQConv2d):
                if module.alpha.shape[0] == 1:
                    writer.add_scalar("Parameter/Alpha/" + name, module.alpha.item(), epoch)
                    writer.add_scalar("Parameter/Delta/" + name, module.delta.item(), epoch)

        print("N bits ", n_bits)

        writer.add_scalar('N bits', n_bits, epoch)
        writer.add_scalar('Top1', prec1, epoch)

        current_df = pd.DataFrame([[epoch, prec1, n_bits]], columns=columns)
        df_log = df_log.append(current_df)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1

        if is_best:
            best_prec1 = prec1
            best_epoch = epoch
            best_bits = n_bits

        print("Current Top1 {:.4f}\t Best Top1 {:.4f}".format(prec1, best_prec1))
        print("Unique params")
        unique_params(model)
        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
        model_optimizer = optimizer.state_dict()
        model_scheduler = lr_scheduler.state_dict()

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.arch,
            'state_dict': model_state_dict,
            'best_prec1': best_prec1,
            'optimizer': model_optimizer,
            'lr_scheduler': model_scheduler,
            'optimizer_q': optimizer_q,
            'lr_scheduler_q': lr_scheduler_q,
            'scaler': scaler

        }, is_best, path=log_dir)


    csv_path = os.path.join(log_dir, "log.csv")
    print("Log file saved to " + csv_path)
    df_log.to_csv(csv_path)
    f = open(csv_path, 'a+')
    f.write(str(args))
    f.write("\n")
    f.write("Best model at epoch {} \t Top1: {:.4f}\t Bits: {:.4f}"
            .format(best_epoch, best_prec1, best_bits))
    writer.close()
    return best_prec1, best_bits


def train_dali(train_loader, model, criterion, optimizer, scaler, optimizer_q=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    bar = Bar('Processing', max=len(train_loader))
    end = time.time()
    for i, batch_data in enumerate(train_loader):
        input = batch_data[0]['data']
        target = batch_data[0]['label'].squeeze().long()

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = input.cuda(non_blocking=True)
        target_var = target

        if scaler:
            with torch.cuda.amp.autocast():
                output = model(input_var)
                loss = criterion(output, target_var)
        else:
            output = model(input_var)
            loss = criterion(output, target_var)

        if optimizer_q:
            optimizer_q.zero_grad()
        optimizer.zero_grad()

        if scaler:
            scaler.scale(loss)

        loss.backward()


        # if scaler:
        #     scaler.step(optimizer)
        #     scaler.update(optimizer)
        # else:
        optimizer.step()


        if optimizer_q:
            optimizer_q.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Top1: {top1:.4f}'.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg

        )
        bar.next()

    bar.finish()


def train_kd(train_loader, model_student, model_teacher, criterion, optimizer, scaler=None, optimizer_q=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model_student.train()
    model_teacher.eval()
    bar = Bar('Processing', max=len(train_loader))
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = input.cuda(non_blocking=True)
        target_var = target

        if scaler:
            with torch.cuda.amp.autocast():
                logits_student = model_student(input_var)
                with torch.no_grad():
                    logits_teacher = model_teacher(input_var)
                loss = criterion(logits_student, logits_teacher)
        else:
            logits_student = model_student(input_var)
            with torch.no_grad():
                logits_teacher = model_teacher(input_var)
            loss = criterion(logits_student, logits_teacher)
        optimizer.zero_grad()

        # compute gradient and do SGD step
        if optimizer_q:
            optimizer_q.zero_grad()

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss)

        loss.backward()

        # if scaler:
        #     scaler.step(optimizer)
        #     scaler.update(optimizer)
        # else:
        optimizer.step()
        output = logits_student.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Top1: {top1:.4f}'.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg

        )
        bar.next()
    bar.finish()


def train(train_loader, model, criterion, optimizer, scaler=None, optimizer_q=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    bar = Bar('Processing', max=len(train_loader))
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = input.cuda(non_blocking=True)
        target_var = target

        if scaler:
            with torch.cuda.amp.autocast():
                output = model(input_var)
                loss = criterion(output, target_var)
        else:
            output = model(input_var)
            loss = criterion(output, target_var)

        # compute gradient and do SGD step
        if optimizer_q:
            optimizer_q.zero_grad()

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss)

        loss.backward()

        # if scaler:
        #     scaler.step(optimizer)
        #     scaler.update(optimizer)
        # else:
        optimizer.step()
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Top1: {top1:.4f}'.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg

        )
        bar.next()
    bar.finish()


def validate(val_loader, model, criterion, scaler=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    bar = Bar('Processing', max=len(val_loader))
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(input_var)
                    loss = criterion(output, target_var)
            else:
                output = model(input_var)
                loss = criterion(output, target_var)

            output = output.float()
            if scaler:
                scaler.scale(loss)

            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Top1: {top1:.4f}'.format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg

            )
            bar.next()

        bar.finish()

    return top1.avg


def validate_dali(val_loader, model, criterion, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    bar = Bar('Processing', max=len(val_loader))
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            input = batch_data[0]['data']
            target = batch_data[0]['label'].squeeze().long()
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(input_var)
                    loss = criterion(output, target_var)
            else:
                output = model(input_var)
                loss = criterion(output, target_var)

            output = output.float()

            if scaler:
                scaler.scale(loss)

            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Top1: {top1:.4f}'.format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg

            )
            bar.next()
        bar.finish()

    return top1.avg


if __name__ == '__main__':
    args = get_parser().parse_args()

    train_model(args)
