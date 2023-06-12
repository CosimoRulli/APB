import argparse
import models_cifar
import models_imagenet


def get_parser():
    model_names_cifar = sorted(name for name in models_cifar.__dict__
                               if name.islower() and not name.startswith("__")
                               and callable(models_cifar.__dict__[name]))
    model_names_imgnet = sorted(name for name in models_imagenet.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models_imagenet.__dict__[name]))

    model_names = model_names_cifar + model_names_imgnet

    training_args = parser = argparse.ArgumentParser(description='Automatic Pruned Quantization for Image Classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    training_args.add_argument_group("Training Args")

    training_args.add_argument('--dataset', '-d',
                        default="cifar10",
                        choices=["cifar10", "imagenet"],
                        help="Dataset name")

    training_args.add_argument('--image-dir',
                        type=str,
                        default="./imagenet",
                        help="Path to imagenet dir (not required for CIFAR")

    training_args.add_argument('--arch', '-a',
                        metavar='ARCH',
                        default='resnet20',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) )

    training_args.add_argument('-j', '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')

    training_args.add_argument('--epochs',
                        default=100,
                        type=int,
                        help='number of total epochs to run')

    training_args.add_argument('-b', '--batch-size',
                        default=128,
                        type=int,
                        help='mini-batch size (default: 128)')

    training_args.add_argument('--lr', '--learning-rate',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate')

    training_args.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')

    training_args.add_argument('--weight-decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')

    training_args.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint')

    training_args.add_argument('-e', '--evaluate',
                        default=False,
                        action='store_true',
                        help='evaluate model on validation set')
    training_args.add_argument('--optim', default="SGD", type=str, choices=("SGD", "Adam"))
    training_args.add_argument('--scheduler', default="cos", type=str, choices=["cos", "mstep"])
    training_args.add_argument('--warmup_epochs', type=int, default=0)

    training_args.add_argument('--kd', default=False, action="store_true", help="use resnet34 as teacher")

    apb_args = parser.add_argument_group("Automatic Prune Binarization Args")

    apb_args.add_argument('--freezing-epoch-alpha-delta', default=-1, type=int)
    apb_args.add_argument('--freezing-epoch-mask', default=-1, type=int)
    apb_args.add_argument('--weight-decay-apq', default=0, type=float)
    apb_args.add_argument('--lr-apq', default=-1, type=float)
    apb_args.add_argument("--re_init_weight", default=False, action="store_true")

    apb_args.add_argument_group("Activation Quantization Args ")

    apb_args.add_argument('--act-quantization-mode', default=None)
    apb_args.add_argument('--optimizer_q', type=str, default='Adam', choices=("SGD", "Adam"),
                         help='optimizer for quantizer paramters')
    apb_args.add_argument('--lr_q', type=float, default=1e-5, help='learning rate for quantizer parameters')
    apb_args.add_argument('--update_scales_every', type=int, default=1, help='update interval in terms of epochs')

    hardware_args = parser.add_argument_group("Hardware Configuration Args ")

    hardware_args.add_argument('--amp', default=False, action="store_true")
    hardware_args.add_argument('--dali', action="store_true", default=False)
    # parser.add_argument('--full_binarization', default=False, action="store_true")
    # parser.add_argument('--start_apb', type=int, default=200, help='Start apb (only full binarization)')
    hardware_args.add_argument('--amp_vali', default=False, action="store_true")
    hardware_args.add_argument(
        '--gpus',
        default='0',
        help='gpus used for training - e.g 0,1,2,3')

    log_args = parser.add_argument_group("Logging Args ")
    log_args.add_argument('--name',
                        type=str,
                        help="The name of the directory where the model will be saved")
    log_args.add_argument('--out-dir', '-o',
                        dest='output_dir',
                        default='logs',
                        help='Path to dump logs and checkpoints')



    return parser