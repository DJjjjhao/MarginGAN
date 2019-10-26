# Thank the authors of pytorch-generative-model-collections and examples of pytorch.
# The github address is https://github.com/znxlwm/pytorch-generative-model-collections
# and https://github.com/pytorch/examples/blob/master/mnist/main.py respectively.
# Our code is widely adapted from their repositories.

import argparse, os, torch
from MarginGAN import MarginGAN


def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='MarginGAN',
                        choices=['MarginGAN'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'],
                        help='The name of dataset')
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--lrC', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--num_labels', type=int, default=100)
    parser.add_argument('--index', type=int, default=1)
    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

        # declare instance for GAN
    if args.gan_type == 'MarginGAN':
        gan = MarginGAN(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

        # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")

    # visualize learned generator
    gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")


if __name__ == '__main__':
    main()
