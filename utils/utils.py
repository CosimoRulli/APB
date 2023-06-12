import numpy as np
from PIL import Image

import torch
import torchvision.datasets as datasets

#from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
#from nvidia.dali.plugin.pytorch import DALIClassificationIterator

#import nvidia.dali.ops as ops
#import nvidia.dali.types as types

import math
import os
import shutil

from torch.nn.modules import loss
import torch.nn.functional as F
import torch.nn as nn
import wandb

from quantization.weight.AutomaticPruneQuantizer import APQConv2d
#from AutomaticPruneQuantization_fullbin import APQConv2d_fullbin

'''
def fast_collate(batch, memory_format=torch.contiguous_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

'''
#lighting data augmentation
imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}




class Lighting(object):
    def __init__(self, alphastd=0.1,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_dataloaders(args):
    print(f"Loading Dataset {args.dataset}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.dataset == "cifar10":

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset == "cifar100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset == "imagenet":
        crop = 224
        val_size = 256
        assert args.image_dir


        if args.dali:
            local_rank = 0
            world_size = 1
            pip_train = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=local_rank,
                                        data_dir=args.image_dir + '/train',
                                        crop=crop, world_size=world_size, local_rank=local_rank)
            pip_train.build()
            train_loader = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size,
                                                         auto_reset=True)

            pip_val = HybridValPipe(batch_size=128, num_threads=args.workers, device_id=local_rank,
                                    data_dir=args.image_dir + '/val',
                                    crop=crop, size=val_size, world_size=world_size, local_rank=local_rank)
            pip_val.build()
            val_loader = DALIClassificationIterator(pip_val, size=pip_val.epoch_size("Reader") // world_size,
                                                       auto_reset=True)

        else:

            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(crop, scale=(0.08, 1.25)),
                Lighting(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])


            val_transform = transforms.Compose([
                transforms.Resize(val_size),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
                normalize,
            ])


            train_dataset = datasets.ImageFolder(args.image_dir + '/train', train_transform)
            val_dataset = datasets.ImageFolder(args.image_dir + '/val', val_transform)




            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers,
                                                       pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False,
                                                     num_workers=args.workers,
                                                     pin_memory=True)

    else:
        raise ValueError("Wrong dataset name")

    return train_loader, val_loader

def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False, rank=None):
    if rank:
        filename = filename + "_rank_"+str(rank)
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(
            path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def print_stats_bireal(model):
    full_precision_params = 0
    csr_params = 0
    binary_params = 0
    alpha = -100
    delta = -100
    max_el = -1

    for name, param in model.state_dict().items():
        if "binary_conv" in name:
            if "alpha" in name:
                alpha = param
            if "delta" in name:
                delta = param
            if "weight" in name:
                w = param
            if "quant_param" in name:
                m = torch.tensor([1])
                mask = torch.gt(torch.tensor([m], device="cuda"), torch.abs((w.abs() - alpha) / delta))
                q_param = torch.where(mask, torch.sign(w) * alpha, w)
                q_param = torch.where(q_param == 0, torch.ones_like(q_param) * alpha, q_param)
                non_bin = (torch.abs(q_param) != alpha).sum()
                pruned_percentage = non_bin / q_param.numel()
                csr_params += non_bin
                binary_params += q_param.numel() - non_bin
                print("{}\t#full precision vals: {}\tsurviving: {:.3f}%\talpha: {:.4f}\tdelta: {:.4f}"
                      .format('.'.join(name.split(".")[1:-1]), non_bin, pruned_percentage*100, alpha.item(), delta.item()))
        else:
            full_precision_params += param.numel()
        max_el = max(max_el, param.numel())
    n_bits = full_precision_params * 32 + csr_params * (32 + int(math.log2(max_el) + 1)) + binary_params
    n_bits = n_bits.item()
    return n_bits


def print_stats(model, args, epoch):
    full_precision_params = 0
    csr_params = 0
    binary_params = 0
    no_compressed = []
    if args.arch.startswith("resnet20"):
        no_compressed = ["conv1", "linear"]
    if args.arch == "vgg_small":
        no_compressed = ["conv0", "fc"]
    if args.arch.startswith("resnet18") or args.arch.startswith("resnet34"):
        no_compressed = ["conv1", "fc"]


    #assert len(no_compressed) != 0

    alpha = -100
    delta = -100

    max_el = -1

    if len(args.gpus) > 1:
        recompute = True
    else:
        recompute = False
    print("Recompute", recompute)
    q_param = None
    w = None
    for name, param in model.state_dict().items():
        if name.split(".")[1].split(".")[0] in no_compressed:
            full_precision_params += param.numel()
        else:
            if "weight" in name and "conv" in name:
                w = param
            if "quant_param" in name:
                if recompute:
                    mask = torch.gt(torch.tensor([1.], device="cuda"), torch.abs((w.abs() - alpha) / delta))
                    q_param = torch.where(mask, torch.sign(w) * alpha, w)
                    q_param = torch.where(q_param == 0, torch.ones_like(q_param) * alpha, q_param)
                else:
                    q_param = param
                non_bin = (torch.abs(q_param) != alpha).sum()
                pruned_percentage = non_bin / q_param.numel()
                csr_params += non_bin
                binary_params += q_param.numel() - non_bin
                print("{}\t#full precision vals: {}\tsurviving: {:.3f}%\talpha: {:.4f}\tdelta: {:.4f}"
                      .format('.'.join(name.split(".")[1:-1]), non_bin, pruned_percentage*100, alpha, delta))
            if "alpha" in name:
                if param.shape[0] == 1:
                    alpha = param.item()
                else:
                    alpha = param[0].item()
            if "delta" in name:
                if param.shape[0] == 1:
                    delta = param.item()
                else:
                    delta = param[0].item()
            elif not "conv" in name:
                full_precision_params += param.numel()
        max_el = max(max_el, param.numel())

    n_bits = full_precision_params * 32 + csr_params * (32 + int(math.log2(max_el) + 1)) + binary_params
    n_bits = n_bits.item()
    return n_bits


def print_stats_wandb(model, args, epoch):
    full_precision_params = 0
    csr_params = 0
    binary_params = 0
    no_compressed = []
    if args.arch.startswith("resnet20"):
        no_compressed = ["conv1", "linear"]
    if args.arch == "vgg_small":
        no_compressed = ["conv0", "fc"]
    if args.arch.startswith("resnet18") or args.arch.startswith("resnet34"):
        no_compressed = ["conv1", "fc"]

    assert len(no_compressed) != 0

    alpha = -100
    delta = -100

    max_el = -1

    if len(args.gpus) > 1:
        recompute = True
    else:
        recompute = False
    print("Recompute", recompute)
    q_param = None
    w = None
    for name, param in model.state_dict().items():
        if name.split(".")[1].split(".")[0] in no_compressed:
            full_precision_params += param.numel()
        else:
            if "weight" in name and "conv" in name:
                w = param
            if "quant_param" in name:
                if recompute:
                    mask = torch.gt(torch.tensor([args.m], device="cuda"), torch.abs((w.abs() - alpha) / delta))
                    q_param = torch.where(mask, torch.sign(w) * alpha, w)
                    q_param = torch.where(q_param == 0, torch.ones_like(q_param) * alpha, q_param)
                else:
                    q_param = param
                non_bin = (torch.abs(q_param) != alpha).sum()
                pruned_percentage = non_bin / q_param.numel()
                csr_params += non_bin
                binary_params += q_param.numel() - non_bin
                wandb.log("model/sparsity/"+name.split(".")[1:-1], pruned_percentage)
                wandb.log("model/nonbin/"+name.split(".")[1:-1], non_bin)
                wandb.log("model/alpha/"+name.split(".")[1:-1], alpha)
                wandb.log("model/delta/"+name.split(".")[1:-1], delta)
                print("{}\t#full precision vals: {}\tsurviving: {:.3f}%\talpha: {:.4f}\tdelta: {:.4f}"
                      .format('.'.join(name.split(".")[1:-1]), non_bin, pruned_percentage*100, alpha, delta))
            if "alpha" in name:
                if param.shape[0] == 1:
                    alpha = param.item()
                else:
                    alpha = param[0].item()
            if "delta" in name:
                if param.shape[0] == 1:
                    delta = param.item()
                else:
                    delta = param[0].item()
            elif not "conv" in name:
                full_precision_params += param.numel()
        max_el = max(max_el, param.numel())

    n_bits = full_precision_params * 32 + csr_params * (32 + int(math.log2(max_el) + 1)) + binary_params
    n_bits = n_bits.item()
    return n_bits


def freeze_alpha_delta(model):
    for module in model.modules():
        if isinstance(module, APQConv2d):# or isinstance(module, APQConv2d_fullbin):
            module.alpha.requires_grad = False
            module.delta.requires_grad = False


def freeze_mask(model):
    for module in model.modules():
        if isinstance(module, APQConv2d):# or isinstance(module, APQConv2d_fullbin):
            module.freeze_mask()
            #module.print_devices()

def unique_params(model):
    for module in model.modules():
        if isinstance(module, APQConv2d):# or isinstance(module, APQConv2d_fullbin):
            module.count_unique_params()

# def start_apb(model, args):
#     for module in model.modules():
#         if isinstance(module, APQConv2d_fullbin):
#             module.full_binarization = False
#             module.init_alpha_delta(args.ignore_m)


def update_irnet_params(epochs, epoch, model):
    T_min, T_max = 1e-1, 1e1

    def Log_UP(K_min, K_max, epoch):
        Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
        return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / epochs * epoch)]).float().cuda()

    t = Log_UP(T_min, T_min, epoch)
    if (t < 1):
        k = 1 / t
    else:
        k = torch.tensor([1]).float().cuda()

    for module in model.modules():
        if isinstance(module, APQConv2d):
            module.activation_quantizer.k = k
            module.activation_quantizer.t = t




class DistributionLoss(loss._Loss):
    def forward(self, model_output, real_output):
        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if real_output.requires_grad:
            raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
