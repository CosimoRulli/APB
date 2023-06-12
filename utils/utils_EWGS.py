import numpy as np
import torch
import sys
from tqdm import tqdm
from quantization.activation.ActivationQuantization import EWGSActivationQuantizer
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def load_imagenet(args):
    print("Reloading imagenet")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop = 224
    # if args.dali:
    #     local_rank = 0
    #     world_size = 1
    #     pip_train = HybridTrainPipe(batch_size=16, num_threads=1, device_id=local_rank,
    #                                 data_dir=args.image_dir + '/train',
    #                                 crop=crop, world_size=world_size, local_rank=local_rank)
    #     pip_train.build()
    #     train_loader = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size,
    #                                               auto_reset=True)
    #
    #
    # else:

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop, scale=(0.08, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])



    train_dataset = datasets.ImageFolder(args.image_dir + '/train', train_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    return train_loader

def is_instance(m):
    if isinstance(m, EWGSActivationQuantizer):# and isinstance(m.activation_quantizer, EWGSActivationQuantizer):
        return True
    return False



def update_grad_scales(model, train_loader, criterion, device, args):
    ## update scales
    
    if args.dataset == "imagenet":
        train_loader = load_imagenet(args) 


    scaleA = []

    for m in model.modules():
        if is_instance(m):
            m.hook_Qvalues = True
            scaleA.append(0)


    model.train()
    # if args.dali:
    #     with tqdm(total=10, file=sys.stdout) as pbar:
    #         for num_batches, batch_data in enumerate(train_loader):
    #             images = batch_data[0]['data']
    #             labels = batch_data[0]['label'].squeeze().long()
    #             if num_batches == 10:
    #
    #                 break
    #             images = images.to(device)
    #             labels = labels.to(device)
    #
    #             # forward with single batch
    #             model.zero_grad()
    #             pred = model(images)
    #             loss = criterion(pred, labels)
    #             loss.backward(create_graph=True)
    #
    #             Qact = []
    #             for m in model.modules():
    #                 if is_instance(m):
    #                     Qact.append(m.buff_act)
    #
    #             # update the scaling factor for activations
    #
    #             params = []
    #             grads = []
    #             for i in range(len(Qact)):  # store variable & gradients
    #                 params.append(Qact[i])
    #                 grads.append(Qact[i].grad)
    #
    #             for i in range(len(Qact)):
    #                 trace_hess_A = np.mean(trace(model, [params[i]], [grads[i]], device))
    #                 avg_trace_hess_A = trace_hess_A / params[i].view(-1).size()[0]  # avg trace of hessian
    #                 #todo here 3 is a magic number, maybe stdev in gaussian
    #                 scaleA[i] += (avg_trace_hess_A / (grads[i].std().cpu().item() * 3.0))
    #
    #             # update the scaling factor for weights
    #
    #             pbar.update(1)
    #
    #
    #
    # else:
    with tqdm(total=20, file=sys.stdout) as pbar:
        for num_batches, (images, labels) in enumerate(train_loader):
            if num_batches == 20:  # estimate trace using 3 batches
                break
            images = images.to(device)
            labels = labels.to(device)

            # forward with single batch
            model.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward(create_graph=True)



            Qact = []
            for m in model.modules():
                if is_instance(m):
                    Qact.append(m.buff_act)

            # update the scaling factor for activations

            params = []
            grads = []
            for i in range(len(Qact)):  # store variable & gradients
                params.append(Qact[i])
                grads.append(Qact[i].grad)

            for i in range(len(Qact)):
                trace_hess_A = np.mean(trace(model, [params[i]], [grads[i]], device))
                avg_trace_hess_A = trace_hess_A / params[i].view(-1).size()[0]  # avg trace of hessian
                scaleA[i] += (avg_trace_hess_A / (grads[i].std().cpu().item() * 3.0))

            # update the scaling factor for weights

            pbar.update(1)


    for i in range(len(scaleA)):
        scaleA[i] /= num_batches
        scaleA[i] = np.clip(scaleA[i], 0, np.inf)
    print("\n\nscaleA\n", scaleA)

    print("")

    i = 0
    for m in model.modules():
        if is_instance(m):
            m.bkwd_scaling_factorA.data.fill_(scaleA[i])
            m.hook_Qvalues = False
            i += 1


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv


def trace(model, params, grads, device, maxIter=50, tol=1e-3):
    """
    compute the trace of hessian using Hutchinson's method
    maxIter: maximum iterations used to compute trace
    tol: the relative tolerance
    """

    trace_vhv = []
    trace = 0.

    for i in range(maxIter):
        model.zero_grad()
        v = [
            torch.randint_like(p, high=2, device=device)
            for p in params
        ]
        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        Hv = hessian_vector_product(grads, params, v)
        trace_vhv.append(group_product(Hv, v).cpu().item())
        if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
            return trace_vhv
        else:
            trace = np.mean(trace_vhv)

    return trace_vhv