import argparse
import json
import os


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(description='MMS 2020 Challenge - DASEGAN', formatter_class=SmartFormatter)

parser.add_argument("--gpu", type=str, default="0,1")
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument('--output_dir', type=str, help='Where progress/checkpoints will be saved')

parser.add_argument('--epochs', type=int, default=200, help='Total number epochs for training')
parser.add_argument('--decay_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')

parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')

parser.add_argument('--dataset', type=str, help='Which dataset use')

parser.add_argument('--data_augmentation', type=str, help='Apply data augmentations at train time')
parser.add_argument('--img_size', type=int, default=224, help='Final img squared size')
parser.add_argument('--crop_size', type=int, default=224, help='Center crop squared size')

parser.add_argument('--normalization', type=str, required=True, help='Data normalization method')
parser.add_argument('--add_depth', action='store_true', help='If apply image transformation 1 to 3 channels or not')

parser.add_argument(
    '--mask_reshape_method', type=str, default="", help='How to resize segmentation predictions.',
    choices=['padd', 'resize']
)

# Networks
parser.add_argument('--seg_net', type=str, default='simple_unet', help='Model name for Segmentator')
parser.add_argument('--dis_net', type=str, default='n_layers', help='Model name for Discriminator')
parser.add_argument('--gen_net', type=str, default='resnet_9blocks', help='Model name for Generator')

parser.add_argument('--norm_layer', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discriminator filters in first conv layer')

parser.add_argument('--seg_checkpoint', type=str, default="", help='If there is a segmentator checkpoint to load')
parser.add_argument('--dis_checkpoint', type=str, default="", help='If there is a discriminator checkpoint to load')
parser.add_argument('--gen_checkpoint', type=str, default="", help='If there is a generator checkpoint to load')

# Tasks weights
parser.add_argument('--cycle_coef', type=float, default=0.5)

parser.add_argument('--generated_samples', type=int, default=0, help='Generated samples to save each epoch')

parser.add_argument('--patients_percentage', type=float, default=1, help='Train patients percentage (from 0 to 1)')
parser.add_argument('--rand_histogram_matching', action='store_true', help='Apply random histogram matching')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.output_dir == "":
    assert False, "Please set an output directory"

for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.generated_samples > 0:
        os.makedirs(os.path.join(args.output_dir, "generated_samples"), exist_ok=True)

# https://stackoverflow.com/a/55114771
with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
