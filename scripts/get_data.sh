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