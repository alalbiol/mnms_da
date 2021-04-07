#!/bin/bash

gpu="0,1"
dataset="mms2d_full_unlabeled_ncl"
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

seg_checkpoint="checks/segmentator/tanh_seg/model_resnet18_unet_scratch_last.pt"
dis_checkpoint=""
gen_checkpoint="DASEGAN_EXPERIMENTS/dasegan_v04_2_1_2_dis_u_coef/dis_u_2/checkpoint_epoch20.pt"
# gen_checkpoint=""

img_size=256
crop_size=256
batch_size=16

epochs=80
decay_epoch=60

lr=0.0002

data_augmentation="mms2d"

normalization="negative1_positive1"
mask_reshape_method="padd"

rfield_method="random_maps"  # "random_maps" - "random_atomic"

#cycle_coef=0.5
dis_u_coef=0.0
realfake_coef=0.2
cycle_coef=0.5

model_dir="GENERATOR_${gen_net}_${gen_upsample}_DISCRIMINATOR_${dis_net}_SEGMENTATOR_${seg_net}"
output_dir="results/$dataset/DASEGAN/$model_dir"
output_dir="$output_dir/lr${lr}_cyclecoef${cycle_coef}_realfakecoef${realfake_coef}_disucoef${dis_u_coef}"
output_dir="$output_dir/normalization_${normalization}/da${data_augmentation}/norm_layer_${norm_layer}"

python3 -u dasegan.py --gpu $gpu --seed $seed  --output_dir "$output_dir" \
--epochs $epochs --decay_epoch $decay_epoch --lr $lr --batch_size $batch_size --dataset $dataset \
--data_augmentation $data_augmentation --img_size $img_size --crop_size $crop_size --normalization $normalization \
--add_depth --mask_reshape_method $mask_reshape_method \
--seg_net $seg_net --dis_net $dis_net --gen_net $gen_net --gen_upsample $gen_upsample \
--gen_norm_layer $gen_norm_layer --dis_norm_layer $dis_norm_layer --ngf $ngf --ndf $ndf \
--seg_checkpoint "$seg_checkpoint" --dis_checkpoint "$dis_checkpoint" --gen_checkpoint "$gen_checkpoint" \
--cycle_coef $cycle_coef --realfake_coef $realfake_coef --dis_u_coef $dis_u_coef \
--plot_examples --rfield_method $rfield_method \
--no_dropout --use_original_mask \
--dis_labels_criterion $dis_labels_criterion --dis_realfake_criterion $dis_realfake_criterion

############################################################
python tools/notify.py --msg "DASEGAN - Experiments Finished"
############################################################
