import argparse
import os
import time
def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print-freq', type=int,
                        default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int,
                        default=512, help='batch_size')
    parser.add_argument('--num_workers', type=int,
                        default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=bool, default=True)
    # optimization
    parser.add_argument('--learning_rate', type=float,
                        default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str,
                        default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100'], help='dataset')

    # distillation
    parser.add_argument('--distill', type=str, default='semckd', choices=['semckd'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float,
                        default=1.0, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float,
                        default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0,
                        help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4,
                        help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=256,
                        type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact',
                        type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int,
                        help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float,
                        help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float,
                        help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=1,
                        type=int, choices=[0, 1, 2, 3, 4])

    # transform layers for IRG
    parser.add_argument('--transform_layer_t', nargs='+', type=int, default=[])
    parser.add_argument('--transform_layer_s', nargs='+', type=int, default=[])

    # switch for edge transformation
    parser.add_argument('--no_edge_transform',
                        action='store_true')  # default=false


    parser.add_argument('--deterministic', default=True,
                        help='Make results reproducible')

    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip validation of teacher')

    parser.add_argument('--hkd_initial_weight', default=100,
                        type=float, help='Initial layer weight for HKD method')
    parser.add_argument('--hkd_decay', default=0.7, type=float,
                        help='Layer weight decay for HKD method')
    
    ## params
    parser.add_argument('--redn_frac', default=0.75, type=float,
                        help='input network size reduction factor')    

    opt = parser.parse_args()

    # set the path of model and tensorboard
    opt.save_folder = './output/{}'.format(time.strftime("%Y%m%d_%H%M%S"))

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

args = parse_option()
