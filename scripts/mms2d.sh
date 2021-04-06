#!/bin/bash

# Only download data --> ./scripts/mms2d.sh only_data

# Check if MMs data is available, if not download
if [ ! -d "data/MMs" ]
then
    echo "MMs data not found at 'data' directory. Downloading..."

    curl -O -J https://nextcloud.maparla.duckdns.org/s/psektTSfsaFa6Xr/download
    mkdir -p data
    tar -zxf MMs_Oficial.tar.gz  -C data/
    rm MMs_Oficial.tar.gz

    curl -O -J https://nextcloud.maparla.duckdns.org/s/BqYoWaYbTB9C83m/download
    mkdir -p data
    tar -zxf MMs_Meta.tar.gz  -C data/MMs
    rm MMs_Meta.tar.gz

    [ "$1" == "only_data" ] && exit

    echo "Done!"
else
  echo "MMs data already downloaded!"
  [ "$1" == "only_data" ] && exit
fi


gpu="0,1"
dataset="mms2d"
problem_type="segmentation"

# Available models:
#   -> resnet34_unet_scratch - resnet18_unet_scratch
#   -> small_segmentation_unet - small_segmentation_small_unet
#      small_segmentation_extrasmall_unet - small_segmentation_nano_unet
#   -> resnet18_pspnet_unet - resnet34_pspnet_unet
model="resnet18_unet_scratch"

img_size=224
crop_size=224
batch_size=32

epochs=170
swa_start=130
defrost_epoch=-1

# Available schedulers:
# constant - steps - plateau - one_cycle_lr (max_lr) - cyclic (min_lr, max_lr, scheduler_steps)
scheduler="steps"
lr=0.001
swa_lr=0.00256
# Available optimizers:
# adam - sgd - over9000
optimizer="adam"

# Available data augmentation policies:
# "none" - "random_crops" - "rotations" - "vflips" - "hflips" - "elastic_transform" - "grid_distortion" - "shift"
# "scale" - "optical_distortion" - "coarse_dropout" or "cutout" - "downscale"
data_augmentation="mms2d"

normalization="negative1_positive1"  # reescale - standardize - standardize_full_vol - standardize_phase
mask_reshape_method="padd"  # padd - resize

generated_overlays=0

# Available criterions:
# bce - bce_dice - bce_dice_ac - bce_dice_border - bce_dice_border_ce
#criterion="bce_dice_border_ce"
#weights_criterion="0.5,0.2,0.2,0.2,0.5"
criterion="bce_dice"
weights_criterion="0.4, 0.5, 0.1"

output_dir="results/$dataset/$model/$optimizer/${scheduler}_lr${lr}/${criterion}_weights${weights_criterion}"
output_dir="$output_dir/normalization_${normalization}/da${data_augmentation}"

#  --generated_overlays $generated_overlays --add_depth --rand_histogram_matching
python3 -u train.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
--epochs $epochs --swa_start $swa_start --batch_size $batch_size --defrost_epoch $defrost_epoch \
--scheduler $scheduler --learning_rate $lr --swa_lr $swa_lr --optimizer $optimizer --criterion $criterion \
--normalization $normalization --weights_criterion "$weights_criterion" --data_augmentation $data_augmentation \
--output_dir "$output_dir" --metrics iou dice --problem_type $problem_type --mask_reshape_method $mask_reshape_method \
--scheduler_steps 70 100 \
--evaluate

: '
model_checkpoint="$output_dir/model_${model}_best_iou.pt"
eval_dir="$output_dir/RESULTS"
python3 -u predict.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
--batch_size $batch_size --normalization $normalization --output_dir "$eval_dir" \
--problem_type $problem_type --mask_reshape_method $mask_reshape_method --metrics iou dice \
--generated_overlays $generated_overlays --add_depth --model_checkpoint "$model_checkpoint"

python3 -u tools/metrics_mnms.py --GT_IMG "data/MMs/Testing" --PRED_IMG "$eval_dir/test_predictions"  # --REMOVE_PREDS


swa_total_epochs="$((epochs-swa_start))"
model_checkpoint="$output_dir/model_${model}_${swa_total_epochs}epochs_swalr${swa_lr}.pt"
eval_dir="$output_dir/SWA_RESULTS"
python3 -u predict.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
--batch_size $batch_size --normalization $normalization --output_dir "$eval_dir" \
--problem_type $problem_type --mask_reshape_method $mask_reshape_method --metrics iou dice \
--generated_overlays $generated_overlays --add_depth --model_checkpoint "$model_checkpoint" --swa_checkpoint


python3 -u tools/metrics_mnms.py --GT_IMG "data/MMs/Testing" --PRED_IMG "$eval_dir/test_predictions"  # --REMOVE_PREDS
'

##################################################
python tools/notify.py --msg "Experiments Finished"
##################################################
