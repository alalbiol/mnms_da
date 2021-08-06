#!/bin/bash

gpu="0,1"
seed=42

dataset="mms2d_full_unlabeled_ncl"  # mms2d_unlabeled_ncl - mms2d_full_unlabeled_ncl

epochs=60
decay_epoch=40
segmentator_lr=0.00001
generator_lr=0.001
discriminator_lr=0.001
batch_size=8

data_augmentation="mms2d"

data_sampling="random_sampler"

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

#seg_checkpoint=""
dasegan_checkpoint=""
seg_checkpoint="checks/segmentator/tanh_seg_256_singlec/model.pt"
dis_checkpoint=""
gen_checkpoint=""

img_size=256
crop_size=256

normalization="negative1_positive1"
mask_reshape_method="padd"

rfield_method="random_maps"  # "random_maps" - "random_atomic"

task_criterion="bce_dice"
task_weights_criterion="0.4, 0.5, 0.1"

for task_loss_u_coef in 0.9
do

for cycle_coef in 2.75
do

for dis_u_coef in 0.25
do

for realfake_coef in 1.0
do

for vendor_label_coef in 0.5
do


unique_id=$(uuidgen)

# --- GENERATOR TRAINING ---

model_dir="GENERATOR_${gen_net}_${gen_upsample}_DISCRIMINATOR_${dis_net}_SEGMENTATOR_${seg_net}"
output_dir="results/$dataset/DASEGAN/$model_dir/normalization_${normalization}"
output_dir="$output_dir/${data_sampling}/da${data_augmentation}/norm_layer_${gen_norm_layer}"
output_dir="$output_dir/seglr${segmentator_lr}_genlr${generator_lr}_dislr${discriminator_lr}"
output_dir="${output_dir}_tasklossucoef${task_loss_u_coef}_cyclecoef${cycle_coef}_realfakecoef${realfake_coef}"
output_dir="${output_dir}_disucoef${dis_u_coef}_vendorlabelcoef${vendor_label_coef}"

python3 -u dasegan.py --gpu $gpu --seed $seed  --output_dir "$output_dir" \
--epochs $epochs --decay_epoch $decay_epoch --batch_size $batch_size --dataset $dataset \
--segmentator_lr $segmentator_lr --generator_lr $generator_lr --discriminator_lr $discriminator_lr \
--data_augmentation $data_augmentation --img_size $img_size --crop_size $crop_size --normalization $normalization \
--mask_reshape_method $mask_reshape_method \
--seg_net $seg_net --dis_net $dis_net --gen_net $gen_net --gen_upsample $gen_upsample \
--gen_norm_layer $gen_norm_layer --dis_norm_layer $dis_norm_layer --ngf $ngf --ndf $ndf \
--seg_checkpoint "$seg_checkpoint" --dis_checkpoint "$dis_checkpoint" --gen_checkpoint "$gen_checkpoint" \
--cycle_coef $cycle_coef --realfake_coef $realfake_coef --dis_u_coef $dis_u_coef --task_loss_u_coef $task_loss_u_coef \
--vendor_label_coef $vendor_label_coef --plot_examples --rfield_method $rfield_method \
--no_dropout --unique_id "$unique_id" --data_sampling $data_sampling \
--dis_labels_criterion $dis_labels_criterion --dis_realfake_criterion $dis_realfake_criterion \
--task_criterion $task_criterion --task_weights_criterion "$task_weights_criterion" --evaluate


done

done

done

done

done

############################################################
python tools/notify.py --msg "DASEGAN - Experiments Finished"
############################################################
