#!/usr/bin/env python
# coding: utf-8
"""
Usage: python tools/testgt2phases.py
"""
import os
import re
import glob
import pandas as pd
import nibabel as nib
from tqdm import tqdm


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_fdata().squeeze(), nimg.affine, nimg.header


def save_nii(img_path, data, affine, header):
    """
    Function to save a 'nii' or 'nii.gz' file.

    Parameters
    ----------

    img_path: string
    Path to save the image should be ending with '.nii' or '.nii.gz'.

    data: np.array
    Numpy array of the image data.

    affine: list of list or np.array
    The affine transformation to save with the image.

    header: nib.Nifti1Header
    The header that define everything about the data
    (pleasecheck nibabel documentation).
    """
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    """
    Sort a (list,tuple) of strings into natural order.

    Ex:

    ['1','10','2'] -> ['1','2','10']

    ['abc1def','ab10d','b2c','ab1d'] -> ['ab1d','ab10d', 'abc1def', 'b2c']

    """
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


info = pd.read_csv(os.path.join("data/MMs", '211230_M&Ms_Dataset_information_diagnosis_opendataset.csv'))
dir_gt = "data/MMs/Testing/"
lst_gt = sorted(glob.glob(os.path.join(dir_gt, '**', '*sa_gt.nii.gz'), recursive=True), key=natural_order)

print("Procesing...")
for vol_path in tqdm(lst_gt, desc="Remaining Files"):
    data, affine, header = load_nii(vol_path)
    external_code = vol_path.split("/")[-2]
    df_entry = info.loc[info["External code"] == external_code]
    ed_volume = data[..., df_entry["ED"]].squeeze()
    es_volume = data[..., df_entry["ES"]].squeeze()

    save_nii(vol_path.replace("sa_gt", "sa_ED_gt"), ed_volume, affine, header)
    save_nii(vol_path.replace("sa_gt", "sa_ES_gt"), es_volume, affine, header)
print("Done!")
