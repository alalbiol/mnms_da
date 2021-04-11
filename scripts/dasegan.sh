#!/bin/bash

gpu="0,1"
seed=42

seg_net="resnet18_unet_scratch"
gen_net="my_resnet_9blocks"
dis_net="n_layers_spectral"
gen_upsample="interpolation"

dis_labels_criterion="ce"
dis_realfake_criterion="bce"

gen_norm_layer="instance" # instance - batch
dis_norm_layer="instance" # instance - batch

ngf=64
ndf=64

seg_checkpoint="checks/segmentator/tanh_seg_256_singlec/model.pt"
dis_checkpoint=""
gen_checkpoint=""

img_size=256
crop_size=256

normalization="negative1_positive1"
mask_reshape_method="padd"

rfield_method="random_maps"  # "random_maps" - "random_atomic"

problem_type="segmentation"

for cycle_coef in 0.1 0.5
do

for dis_u_coef in 0.5 2.0
do

for realfake_coef in 0.5 2.0
do

for vendor_label_coef in 0.5 2.0
do


unique_id=$(uuidgen)

# --- GENERATOR TRAINING ---

dataset="mms2d_full_unlabeled_ncl"  # mms2d_unlabeled_ncl - mms2d_full_unlabeled_ncl

epochs=60
decay_epoch=40
lr=0.001
batch_size=16

data_augmentation="mms2d"

model_dir="GENERATOR_${gen_net}_${gen_upsample}_DISCRIMINATOR_${dis_net}_SEGMENTATOR_${seg_net}"
output_dir="results/$dataset/DASEGAN/$model_dir"
output_dir="$output_dir/lr${lr}_cyclecoef${cycle_coef}_realfakecoef${realfake_coef}_disucoef${dis_u_coef}_vendorlabelcoef${vendor_label_coef}"
output_dir="$output_dir/normalization_${normalization}/da${data_augmentation}/norm_layer_${gen_norm_layer}"

python3 -u dasegan.py --gpu $gpu --seed $seed  --output_dir "$output_dir" \
--epochs $epochs --decay_epoch $decay_epoch --lr $lr --batch_size $batch_size --dataset $dataset \
--data_augmentation $data_augmentation --img_size $img_size --crop_size $crop_size --normalization $normalization \
--mask_reshape_method $mask_reshape_method \
--seg_net $seg_net --dis_net $dis_net --gen_net $gen_net --gen_upsample $gen_upsample \
--gen_norm_layer $gen_norm_layer --dis_norm_layer $dis_norm_layer --ngf $ngf --ndf $ndf \
--seg_checkpoint "$seg_checkpoint" --dis_checkpoint "$dis_checkpoint" --gen_checkpoint "$gen_checkpoint" \
--cycle_coef $cycle_coef --realfake_coef $realfake_coef --dis_u_coef $dis_u_coef \
--vendor_label_coef $vendor_label_coef --plot_examples --rfield_method $rfield_method \
--no_dropout --use_original_mask --unique_id "$unique_id" \
--dis_labels_criterion $dis_labels_criterion --dis_realfake_criterion $dis_realfake_criterion


# --- SEGMENTATION TRAINING WITH GENERATOR ---

dataset="mms2d"

epochs=170
swa_start=130
defrost_epoch=-1

model="resnet18_unet_scratch"

scheduler="steps"
lr=0.001
swa_lr=0.00256
batch_size=4
optimizer="adam"

data_augmentation="mms2d"

criterion="bce_dice"
weights_criterion="0.4, 0.5, 0.1"

gen_checkpoint="$output_dir/checkpoint.pt"
output_dir="$output_dir/SEGMENTATION"

python3 -u train.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
--epochs $epochs --swa_start $swa_start --batch_size $batch_size --defrost_epoch $defrost_epoch \
--scheduler $scheduler --learning_rate $lr --swa_lr $swa_lr --optimizer $optimizer --criterion $criterion \
--normalization $normalization --weights_criterion "$weights_criterion" --data_augmentation $data_augmentation \
--output_dir "$output_dir" --metrics iou dice --problem_type $problem_type --mask_reshape_method $mask_reshape_method \
--scheduler_steps 70 100 --gen_net $gen_net --gen_upsample $gen_upsample --norm_layer $gen_norm_layer --ngf $ngf \
--no_dropout --gen_checkpoint $gen_checkpoint --unique_id "$unique_id" --evaluate

done

done

done

done

############################################################
python tools/notify.py --msg "DASEGAN - Experiments Finished"
############################################################
