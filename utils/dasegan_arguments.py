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

parser.add_argument('--epochs', type=int, default=80, help='Total number epochs for training')
parser.add_argument('--decay_epoch', type=int, default=60)
parser.add_argument('--segmentator_lr', type=float, default=0.0002, help='Learning rate')
parser.add_argument('--generator_lr', type=float, default=0.0002, help='Learning rate')
parser.add_argument('--discriminator_lr', type=float, default=0.0002, help='Learning rate')

parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')

parser.add_argument('--dataset', type=str, help='Which dataset use')

parser.add_argument('--use_original_mask', action='store_true', help='Whether use original mask labels when available')

parser.add_argument('--data_augmentation', type=str, help='Apply data augmentations at train time')
parser.add_argument('--img_size', type=int, default=256, help='Final img squared size')
parser.add_argument('--crop_size', type=int, default=256, help='Center crop squared size')

parser.add_argument('--normalization', type=str, required=True, help='Data normalization method')
parser.add_argument('--add_depth', action='store_true', help='If apply image transformation 1 to 3 channels or not')

parser.add_argument(
    '--mask_reshape_method', type=str, default="", help='How to resize segmentation predictions.',
    choices=['padd', 'resize']
)

parser.add_argument('--weighted_sampler', action='store_true', help='If apply weighted sampling or not')

# Networks
parser.add_argument('--seg_net', type=str, default='resnet18_unet_scratch', help='Model name for Segmentator')
parser.add_argument('--dis_net', type=str, default='n_layers_spectral', help='Model name for Discriminator')
parser.add_argument('--gen_net', type=str, default='my_resnet_9blocks', help='Model name for Generator')

parser.add_argument('--dis_labels_criterion', type=str, default='ce', help='Loss for vendor labels training')
parser.add_argument('--dis_realfake_criterion', type=str, default='bce', help='Loss for real fake training')
parser.add_argument('--task_criterion', type=str, default='bce', help='Criterion for training')
parser.add_argument('--task_weights_criterion', type=str, default='default', help='Weights for each subcriterion')

parser.add_argument(
    '--rfield_method', type=str, required=True, choices=["random_maps", "random_atomic"],
    help='How to generate maps for generator vendor training from discriminator output (vendor branch)'
)

parser.add_argument('--gen_norm_layer', type=str, default='instance', help='Generator normalization layer type')
parser.add_argument('--dis_norm_layer', type=str, default='instance', help='Discriminator normalization layer type')
parser.add_argument(
    '--gen_upsample', type=str, default="interpolation", help='How to perform upsample steps in generator architecture',
    choices=['interpolation', 'deconvolution']
)
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discriminator filters in first conv layer')

parser.add_argument('--seg_checkpoint', type=str, default="", help='If there is a segmentator checkpoint to load')
parser.add_argument('--dis_checkpoint', type=str, default="", help='If there is a discriminator checkpoint to load')
parser.add_argument('--gen_checkpoint', type=str, default="", help='If there is a generator checkpoint to load')

parser.add_argument(
    '--data_sampling', type=str, default="random_sampler", help='How to sample data points',
    choices=['random_sampler', 'equilibrated_sampler']
)

# Tasks weights
parser.add_argument('--cycle_coef', type=float, default=0.5)
parser.add_argument('--vendor_label_coef', type=float, default=1)
parser.add_argument('--realfake_coef', type=float, default=0.0)
parser.add_argument('--dis_u_coef', type=float, default=0.0)
parser.add_argument('--task_loss_u_coef', type=float, default=0.1)

parser.add_argument('--plot_examples', action='store_true', help='Whether plot examples of transformed volumes')

parser.add_argument('--patients_percentage', type=float, default=1, help='Train patients percentage (from 0 to 1)')
parser.add_argument('--rand_histogram_matching', action='store_true', help='Apply random histogram matching')

parser.add_argument('--unique_id', type=str, required=True, help='Unique identifier for current run')

parser.add_argument('--evaluate', action='store_true', help='Whether evaluate using test partition or not')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.output_dir == "":
    assert False, "Please set an output directory"

for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

os.makedirs(args.output_dir, exist_ok=True)
if args.plot_examples:
    os.makedirs(os.path.join(args.output_dir, "generated_samples"), exist_ok=True)

# https://stackoverflow.com/a/55114771
with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

str_ids = args.gpu.split(',')
args.gpu = []
for str_id in str_ids:
    gpu_id = int(str_id)
    if gpu_id >= 0:
        args.gpu.append(gpu_id)
