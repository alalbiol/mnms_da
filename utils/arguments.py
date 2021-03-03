import argparse
import json
import os


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(description='MMS 2020 Challenge - Training', formatter_class=SmartFormatter)

parser.add_argument("--gpu", type=str, default="0,1")
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument('--output_dir', type=str, help='Where progress/checkpoints will be saved')

parser.add_argument(
    '--problem_type', type=str, default="", help='Deep Learning problem type.',
    choices=['classification', 'segmentation']
)

parser.add_argument('--epochs', type=int, default=150, help='Total number epochs for training')
parser.add_argument('--dataset', type=str, help='Which dataset use')
parser.add_argument('--defrost_epoch', type=int, default=-1, help='Number of epochs to defrost the model')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')
parser.add_argument('--data_augmentation', type=str, help='Apply data augmentations at train time')
parser.add_argument('--rand_histogram_matching', action='store_true', help='Apply random histogram matching')
parser.add_argument('--img_size', type=int, default=224, help='Final img squared size')
parser.add_argument('--crop_size', type=int, default=224, help='Center crop squared size')

parser.add_argument('--normalization', type=str, required=True, help='Data normalization method')
parser.add_argument('--add_depth', action='store_true', help='If apply image transformation 1 to 3 channels or not')

parser.add_argument('--model_name', type=str, default='simple_unet', help='Model name for training')
parser.add_argument('--num_classes', type=int, default=1, help='Model output neurons')

# Accept a list of string metrics: train.py --metrics iou dice hauss
parser.add_argument('--metrics', '--names-list', nargs='+', default=[])

parser.add_argument('--generated_overlays', type=int, default=-1, help='Number of generate masks overlays')

parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for training')
parser.add_argument('--scheduler', type=str, default="", help='Where is the model checkpoint saved')
parser.add_argument('--plateau_metric', type=str, default="", help='Metric name to set plateau')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--min_lr', type=float, default=0.0001, help='Minimun Learning rate')
parser.add_argument('--max_lr', type=float, default=0.01, help='Maximum Learning rate')
parser.add_argument(
    '--scheduler_steps', '--arg', nargs='+', type=int, help='Steps when steps and cyclic scheduler choosed'
)

parser.add_argument('--criterion', type=str, default='bce', help='Criterion for training')
parser.add_argument('--weights_criterion', type=str, default='default', help='Weights for each subcriterion')

parser.add_argument('--coral', action='store_true', help='Whether apply coral loss or not')
parser.add_argument('--coral_vendors', '--argc', nargs='+', type=str, help='Which vendors are used for coral loss')
parser.add_argument('--coral_weight', type=float, default=0.01, help='Coral loss weight')
parser.add_argument('--vol_task_weight', type=float, default=0.7, help='Volume task loss (used when coral)')

parser.add_argument('--model_checkpoint', type=str, default="", help='If there is a model checkpoint to load')
parser.add_argument('--swa_checkpoint', action='store_true', help='If we load the model checkpoint from SWA model')

parser.add_argument('--swa_freq', type=int, default=1, help='SWA Frequency')
parser.add_argument('--swa_start', type=int, default=-1, help='Epoch to start SWA and scheduler SWA_LR')
parser.add_argument('--swa_lr', type=float, default=0.05, help='SWA learning rate scheduler WA_LR')

parser.add_argument(
    '--mask_reshape_method', type=str, default="", help='How to reescale segmentation predictions.',
    choices=['padd', 'resize']
)

parser.add_argument('--selected_class', type=str, default="", help='If there is a model checkpoint to load')

parser.add_argument('--patients_percentage', type=float, default=1, help='Train patients percentage (from 0 to 1)')


parser.add_argument('--gen_net', type=str, default='resnet_9blocks', help='Model name for Generator')
parser.add_argument(
    '--gen_upsample', type=str, default="interpolation", help='How to perform upsample steps in generator architecture',
    choices=['interpolation', 'deconvolution']
)
parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in first conv layer')
parser.add_argument('--norm_layer', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--gen_checkpoint', type=str, default="", help='If there is a generator checkpoint to load')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.output_dir == "":
    assert False, "Please set an output directory"

for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

# https://stackoverflow.com/a/55114771
with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

str_ids = args.gpu.split(',')
args.gpu = []
for str_id in str_ids:
    gpu_id = int(str_id)
    if gpu_id >= 0:
        args.gpu.append(gpu_id)
