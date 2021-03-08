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
dataset="mms2d_full_unlabeled_ncl"
seed=42

seg_net="resnet18_unet_scratch"
dis_net="n_layers"
gen_net="resnet_9blocks"
gen_upsample="interpolation"

norm_layer="instance"

ngf=64
ndf=64

seg_checkpoint="checks/segmentator/model_resnet18_unet_scratch_last.pt"
dis_checkpoint=""
gen_checkpoint=""


img_size=256
crop_size=256
batch_size=16

epochs=200
decay_epoch=100

lr=0.0002

data_augmentation="mms2d"

normalization="standardize"
mask_reshape_method="padd"

generated_samples=25

#cycle_coef=0.5
realfake_coef=0.0
for cycle_coef in 0.5 1.0
do

model_dir="GENERATOR_${gen_net}_${gen_upsample}_DISCRIMINATOR_${dis_net}_SEGMENTATOR_${seg_net}"
output_dir="results/$dataset/DASEGAN/$model_dir/lr${lr}_cyclecoef${cycle_coef}"
output_dir="$output_dir/normalization_${normalization}/da${data_augmentation}/norm_layer_${norm_layer}"

python3 -u dasegan.py --gpu $gpu --seed $seed  --output_dir "$output_dir" \
--epochs $epochs --decay_epoch $decay_epoch --lr $lr --batch_size $batch_size --dataset $dataset \
--data_augmentation $data_augmentation --img_size $img_size --crop_size $crop_size --normalization $normalization \
--add_depth --mask_reshape_method $mask_reshape_method \
--seg_net $seg_net --dis_net $dis_net --gen_net $gen_net --gen_upsample $gen_upsample \
--norm_layer $norm_layer --ngf $ngf --ndf $ndf \
--seg_checkpoint "$seg_checkpoint" --dis_checkpoint "$dis_checkpoint" --gen_checkpoint "$gen_checkpoint" \
--cycle_coef $cycle_coef --realfake_coef $realfake_coef \
--generated_samples $generated_samples \
--no_dropout --use_original_mask

done

############################################################
python tools/notify.py --msg "DASEGAN - Experiments Finished"
############################################################
