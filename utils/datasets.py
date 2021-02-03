import random
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import albumentations
import copy
import pandas as pd
from skimage.exposure import match_histograms

import utils.dataload as data_utils
from tools.metrics_mnms import load_nii
from utils.general import map_mask_classes


class MMs2DDataset(Dataset):
    """
    2D Dataset
    """

    def __init__(self, partition, transform, img_transform, normalization="normalize", add_depth=True,
                 is_labeled=True, centre=None, vendor=None, end_volumes=True, data_relative_path="",
                 only_phase="", rand_histogram_matching=False, patients_percentage=1):
        """
        :param partition: (string) Dataset partition in ["Training", "Validation", "Test"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        :param add_depth: (bool) Whether transform or not 1d slices to 3 channels images
        :param is_labeled: (bool) Dataset partition in ["Training", "Validation", "Test"]
        :param centre: (int) Select by centre label. Available [1, 2, 3, 4, 5]
        :param vendor: (string) Select by vendor label. Available ["A", "B", "C", "D"]
        :param end_volumes: (bool) Whether only include 'ED' and 'ES' phases ((to) segmented) or all
        :param data_relative_path: (string) Prepend extension to MMs data base dir
        :param only_phase: (string) Select only phases by 'ED' or 'ES'
        :param rand_histogram_matching: (bool) Perform random histogram matching with different vendors
        :param patients_percentage: (float) Train patients percentage (from 0 to 1)
        """

        if partition not in ["Training", "Validation", "Testing", "All", "All_val"]:
            assert False, "Unknown mode '{}'".format(partition)

        self.base_dir = os.path.join(data_relative_path, "data/MMs")
        self.partition = partition
        self.img_channels = 3 if add_depth else 1
        self.class_to_cat = {1: "LV", 2: "MYO", 3: "RV", 4: "Mean"}
        self.include_background = False
        self.num_classes = 4  # background - LV - MYO - RV

        data = pd.read_csv(os.path.join(self.base_dir, "slices_info.csv"))
        if "All" not in self.partition:
            data = data.loc[(data["Partition"] == partition)]
            data = data.loc[(data["Labeled"] == is_labeled)]
        else:
            # Training and Unlabeled (vendor C) is not labeled!
            data = data.loc[((data['Partition'] == "Training") & (data['Labeled'])) | (data['Partition'] != "Training")]
        if vendor is not None:
            data = data.loc[data['Vendor'].isin(vendor)]
        if centre is not None:
            data = data.loc[data['Centre'].isin(centre)]

        if end_volumes:  # Get only volumes in 'ED' and 'ES' phases (segmented)
            data = data.loc[(data["ED"] == data["Phase"]) | (data["ES"] == data["Phase"])]

        if only_phase != "":
            if only_phase not in ["ED", "ES"]:
                assert False, f"Only ED and ES phases available (selected '{only_phase}')"
            data = data.loc[(data[only_phase] == data["Phase"])]

        if patients_percentage != 1:
            patient_list = np.sort(data["External code"].unique())
            np.random.seed(1)
            train_indx = np.random.choice(range(len(patient_list)),
                                          size=(int(patients_percentage * len(patient_list)),), replace=False)
            train_patients = patient_list[train_indx]
            data = data.loc[data["External code"].isin(train_patients)]

        data = data.reset_index(drop=True)

        self.data = data
        self.data_meta = pd.read_csv(os.path.join(self.base_dir, "volume_info_statistics.csv"))

        self.add_depth = add_depth
        self.normalization = normalization
        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)

        self.rand_histogram_matching = rand_histogram_matching
        if self.rand_histogram_matching:
            data = pd.read_csv(os.path.join(self.base_dir, "slices_info.csv"))
            self.hist_match_df = data.loc[data["Partition"] == partition] if "All" not in self.partition else data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def custom_collate(batch):
        """

        Args:
            batch: list of dataset items (from __getitem__). In this case batch is a list of dicts with
                   key image, and depending of validation or train different keys

        Returns:

        """
        # We have to modify "original_mask" as has different shapes
        batch_keys = list(batch[0].keys())
        res = {bkey: [] for bkey in batch_keys}
        for belement in batch:
            for bkey in batch_keys:
                res[bkey].append(belement[bkey])

        for bkey in batch_keys:
            if bkey == "original_mask" or bkey == "original_img" or bkey == "img_id":
                continue  # We wont stack over original_mask...
            res[bkey] = torch.stack(res[bkey]) if None not in res[bkey] else None

        return res

    def histogram_matching_augmentation(self, image, original_vendor):
        # 40% of the time perform histogram matching with different vendor slice
        if self.partition in ["Training", "All"] and self.rand_histogram_matching and (random.random() < 0.4):
            rand_hist_entry = self.hist_match_df[self.hist_match_df["Vendor"] != original_vendor].sample(n=1).iloc[0]
            external_code = rand_hist_entry["External code"]
            c_slice = rand_hist_entry["Slice"]
            c_phase = rand_hist_entry["Phase"]
            c_vendor = rand_hist_entry["Vendor"]
            c_partition = rand_hist_entry["Partition"]
            labeled_info = "Unlabeled" if c_vendor == "C" else "Labeled"
            reference_img_path = os.path.join(
                self.base_dir, c_partition, labeled_info, external_code,
                f"{external_code}_sa_slice{c_slice}_phase{c_phase}.npy"
            )
            reference = np.load(reference_img_path)
            image = match_histograms(image, reference, multichannel=False)
        return image

    def __getitem__(self, idx):
        df_entry = self.data.loc[idx]
        external_code = df_entry["External code"]
        c_slice = df_entry["Slice"]
        c_phase = df_entry["Phase"]
        c_partition = df_entry["Partition"]
        if c_phase == df_entry["ED"]:
            c_phase_str = "ED"
        elif c_phase == df_entry["ES"]:
            c_phase_str = "ES"
        else:
            assert False, "Not in ED or ES phases?!"
        c_vendor = df_entry["Vendor"]
        c_centre = df_entry["Centre"]
        meta_entry = self.data_meta.loc[self.data_meta["External code"] == external_code]
        img_id = f"{external_code}_slice{c_slice}_phase{c_phase}_vendor{c_vendor}_centre{c_centre}"

        labeled_info = ""
        if c_partition == "Training":
            labeled_info = "Labeled" if df_entry["Labeled"] else "Unlabeled"

        img_path = os.path.join(
            self.base_dir, c_partition, labeled_info, external_code,
            f"{external_code}_sa_slice{c_slice}_phase{c_phase}.npy"
        )
        image = np.load(img_path)
        original_image = copy.deepcopy(image)

        image = self.histogram_matching_augmentation(image, c_vendor)

        mask = None
        if self.partition == "All" or not (c_partition == "Training" and not df_entry["Labeled"]):
            mask_path = os.path.join(
                self.base_dir, c_partition, labeled_info, external_code,
                f"{external_code}_sa_gt_slice{c_slice}_phase{c_phase}.npy"
            )
            mask = np.load(mask_path)
        original_mask = copy.deepcopy(mask)

        image, mask = data_utils.apply_augmentations(image, self.transform, self.img_transform, mask)

        if self.normalization == "standardize_full_vol":
            mean = float(meta_entry["Vol_mean"])
            std = float(meta_entry["Vol_std"])
            image = data_utils.apply_normalization(image, "standardize", mean=mean, std=std)
        elif self.normalization == "standardize_phase":
            mean = float(meta_entry[f"{c_phase_str}_mean"])
            std = float(meta_entry[f"{c_phase_str}_std"])
            image = data_utils.apply_normalization(image, "standardize", mean=mean, std=std)
        elif self.normalization == "reescale_full_vol":
            phase_max = float(meta_entry[f"Vol_max"])
            phase_min = float(meta_entry[f"Vol_min"])
            image = data_utils.apply_normalization(image, "reescale", image_max=phase_max, image_min=phase_min)
        elif self.normalization == "reescale_phase":
            phase_max = float(meta_entry[f"{c_phase_str}_max"])
            phase_min = float(meta_entry[f"{c_phase_str}_min"])
            image = data_utils.apply_normalization(image, "reescale", image_max=phase_max, image_min=phase_min)
        else:
            image = data_utils.apply_normalization(image, self.normalization)

        image = torch.from_numpy(np.expand_dims(image, axis=0))

        if self.add_depth:
            image = data_utils.add_depth_channels(image)
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float() if mask is not None else None

        return {
            "img_id": img_id, "image": image, "label": mask,
            "original_img": original_image, "original_mask": original_mask
        }


class MMs3DDataset(Dataset):
    """
    2D Dataset
    """

    def __init__(self, partition, transform, img_transform, normalization="normalize", add_depth=True,
                 is_labeled=True, centre=None, vendor=None, end_volumes=True, data_relative_path="",
                 only_phase=""):
        """
        :param partition: (string) Dataset partition in ["Training", "Validation", "Test"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        :param add_depth: (bool) Whether transform or not 1d slices to 3 channels images
        :param is_labeled: (bool) Dataset partition in ["Training", "Validation", "Test"]
        :param centre: (int) Select by centre label. Available [1, 2, 3, 4, 5]
        :param vendor: (string) Select by vendor label. Available ["A", "B", "C", "D"]
        :param end_volumes: (bool) Whether only include 'ED' and 'ES' phases ((to) segmented) or all
        :param data_relative_path: (string) Prepend extension to MMs data base dir
        :param only_phase: (string) Select only phases by 'ED' or 'ES'
        """

        if partition not in ["Training", "Validation", "Testing"]:
            assert False, "Unknown mode '{}'".format(partition)

        self.base_dir = os.path.join(data_relative_path, "data/MMs")
        self.partition = partition
        self.img_channels = 3 if add_depth else 1
        self.class_to_cat = {1: "LV", 2: "MYO", 3: "RV", 4: "Mean"}
        self.include_background = False
        self.num_classes = 4  # background - LV - MYO - RV

        data = pd.read_csv(os.path.join(self.base_dir, "detailed_volume_info.csv"))
        data = data.loc[(data["Partition"] == partition) & (data["Labeled"] == is_labeled)]
        if vendor is not None:
            data = data.loc[data['Vendor'].isin(vendor)]
        if centre is not None:
            data = data.loc[data['Centre'].isin(centre)]

        if only_phase != "":
            if only_phase not in ["ED", "ES"]:
                assert False, f"Only ED and ES phases available (selected '{only_phase}')"
            data = data.loc[data["PhaseType"] == only_phase]

        data = data.reset_index(drop=True)
        self.data = data
        self.data_meta = pd.read_csv(os.path.join(self.base_dir, "volume_info_statistics.csv"))

        self.add_depth = add_depth
        self.normalization = normalization
        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def custom_collate(batch):
        """

        Args:
            batch: list of dataset items (from __getitem__). In this case batch is a list of dicts with
                   key image, and depending of validation or train different keys

        Returns:

        """
        not_stack_items = ["original_mask", "original_volume", "volume_id", "vol_slices", "labeled_info"]
        # We have to modify "original_mask" as has different shapes
        batch_keys = list(batch[0].keys())
        res = {bkey: [] for bkey in batch_keys}
        for belement in batch:
            for bkey in batch_keys:
                res[bkey].append(belement[bkey])

        for bkey in batch_keys:
            if bkey in not_stack_items:
                continue  # We wont stack over original_mask...
            res[bkey] = torch.stack(res[bkey]) if None not in res[bkey] else None

        return res

    def __getitem__(self, idx):
        df_entry = self.data.loc[idx]
        external_code = df_entry["External code"]
        c_phase = df_entry["Phase"]
        c_vendor = df_entry["Vendor"]
        c_centre = df_entry["Centre"]
        volume_id = f"{external_code}_phase{c_phase}_vendor{c_vendor}_centre{c_centre}"

        labeled_info = ""
        if self.partition == "Training":
            labeled_info = "Labeled" if df_entry["Labeled"] else "Unlabeled"

        volume_path = os.path.join(
            self.base_dir, self.partition, labeled_info, external_code,
            f"{external_code}_sa.nii.gz"
        )
        volume, affine, header = load_nii(volume_path)
        vol_mean, vol_std, vol_max, vol_min = volume.mean(), volume.std(), volume.max(), volume.min()
        vol_slices = volume.shape[2]
        volume = volume[..., c_phase].transpose(2, 0, 1)

        mask = None
        if not (self.partition == "Training" and not df_entry["Labeled"]):
            mask_path = os.path.join(
                self.base_dir, self.partition, labeled_info, external_code,
                f"{external_code}_sa_gt.nii.gz"
            )
            mask, mask_affine, mask_header = load_nii(mask_path)
            mask = mask[..., c_phase].transpose(2, 0, 1)

        original_volume = copy.deepcopy(volume)
        original_mask = copy.deepcopy(mask)

        volume, mask = data_utils.apply_volume_2Daugmentations(volume, self.transform, self.img_transform, mask)

        if self.normalization == "standardize_full_vol":
            volume = data_utils.apply_normalization(volume, "standardize", mean=vol_mean, std=vol_std)
        elif self.normalization == "standardize_phase":
            volume = data_utils.apply_normalization(volume, "standardize", mean=volume.mean(), std=volume.std())
        elif self.normalization == "reescale_full_vol":
            volume = data_utils.apply_normalization(volume, "reescale", image_max=vol_max, image_min=vol_min)
        elif self.normalization == "reescale_phase":
            volume = data_utils.apply_normalization(volume, "reescale", image_max=volume.max(), image_min=volume.min())
        else:
            volume = data_utils.apply_normalization(volume, self.normalization)

        # We have to stack volume as batch
        volume = np.expand_dims(volume, axis=0) if not self.add_depth else volume
        volume = torch.from_numpy(volume)

        if self.add_depth:
            volume = data_utils.add_volume_depth_channels(volume.unsqueeze(1))
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float() if mask is not None else None

        return {
            "volume_id": volume_id, "volume": volume, "label": mask, "vol_slices": vol_slices,
            "original_volume": original_volume, "original_mask": original_mask, "labeled_info": labeled_info
        }


def get_volume_loader(vendor, train_aug, train_aug_img, add_depth=True, partition="Training", data_relative_path=""):
    """
    Helper function for easily create data loaders for coral loss application
    """
    normalization = "standardize"
    data_mod = ""

    batch_size = 1

    dataset = f"mms_vendor{vendor}{data_mod}"

    only_end = False if "full" in dataset else True
    unlabeled = True if "unlabeled" in dataset or partition in ["Validation", "Testing"] else False
    c_centre = find_values(dataset, "centre", int)
    c_vendor = find_values(dataset, "vendor", str)

    dataset = MMs3DDataset(
        partition=partition, transform=train_aug, img_transform=train_aug_img, normalization=normalization,
        add_depth=add_depth, is_labeled=(not unlabeled), centre=c_centre, vendor=c_vendor, end_volumes=only_end,
        data_relative_path=data_relative_path
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, collate_fn=dataset.custom_collate,
        shuffle=True if partition == "Training" else False,
    )

    return loader


class MMsSubmissionDataset(Dataset):
    """
    Submission Dataset for Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms).
    """

    def __init__(self, transform, img_transform, add_depth=False, normalization="standardize",
                 data_relative_path=""):
        """
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param add_depth: (bool) If apply image transformation 1 to 3 channels or not
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        self.img_channels = 3 if add_depth else 1
        self.class_to_cat = {1: "LV", 2: "MYO", 3: "RV", 4: "Mean"}
        self.include_background = False
        self.num_classes = 4  # background - LV - MYO - RV
        self.base_dir = os.path.join(data_relative_path, "data/MMs")
        info_csv = os.path.join(self.base_dir, "volume_info.csv")
        if not os.path.exists(info_csv):
            assert False, "Cannot find info.csv at input path {}!".format(self.base_dir)
        data = pd.read_csv(info_csv)
        data = data.loc[data["Partition"] == "Testing"]
        data = data.reset_index(drop=True)
        self.df = data

        self.add_depth = add_depth
        self.normalization = normalization
        self.transform = albumentations.Compose(transform)
        if not img_transform:
            self.img_transform = None
        else:
            self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return len(self.df)

    def apply_volume_augmentations(self, list_images):
        """
        Apply same augmentations to volume images
        :param list_images: (array) [num_images, height, width] Images to transform
        :return: (array) [num_images, height, width] Transformed Images
        """
        if self.img_transform:
            # Independent augmentations...
            for indx, img in enumerate(list_images):
                augmented = self.img_transform(image=img)
                list_images[indx] = augmented['image']

        if self.transform:
            # All augmentations applied in same proportion and values
            imgs_ids = ["image"] + ["image{}".format(idx + 2) for idx in range(len(list_images) - 1)]
            aug_args = dict(zip(imgs_ids, list_images))

            pair_ids_imgs = ["image{}".format(idx + 2) for idx in range(len(list_images) - 1)]
            base_id_imgs = ["image"] * len(pair_ids_imgs)
            list_additional_targets = dict(zip(pair_ids_imgs, base_id_imgs))

            volumetric_aug = albumentations.Compose(self.transform, additional_targets=list_additional_targets)
            augmented = volumetric_aug(**aug_args)

            list_images = np.stack([augmented[img] for img in imgs_ids])

        return list_images

    def apply_volume_normalization(self, list_images, volume):

        if self.normalization == "standardize_full_vol":
            mean = volume.mean()
            std = volume.std()
            list_images = data_utils.apply_normalization(list_images, "standardize", mean=mean, std=std)
        elif self.normalization == "standardize_phase":
            mean = list_images.mean()
            std = list_images.std()
            list_images = data_utils.apply_normalization(list_images, "standardize", mean=mean, std=std)
        elif self.normalization == "reescale_full_vol":
            phase_max = volume.max()
            phase_min = volume.min()
            list_images = data_utils.apply_normalization(list_images, "reescale", image_max=phase_max,
                                                         image_min=phase_min)
        elif self.normalization == "reescale_phase":
            phase_max = list_images.max()
            phase_min = list_images.min()
            list_images = data_utils.apply_normalization(list_images, "reescale", image_max=phase_max,
                                                         image_min=phase_min)
        else:
            for indx, image in enumerate(list_images):
                list_images[indx, ...] = data_utils.apply_normalization(image, self.normalization)
        return list_images

    def simple_collate(self, batch):
        return batch[0]

    def __getitem__(self, idx):

        external_code = self.df.loc[idx]['External code']
        ed_phase = self.df.loc[idx]["ED"]
        es_phase = self.df.loc[idx]["ES"]

        img_path = os.path.join(
            self.base_dir, "Testing", external_code, "{}_sa.nii.gz".format(external_code)
        )
        volume, affine, header = load_nii(img_path)
        ed_volume = volume[..., :, ed_phase]
        es_volume = volume[..., :, es_phase]

        original_ed = copy.deepcopy(ed_volume)
        original_es = copy.deepcopy(es_volume)
        initial_shape = es_volume.shape

        ed_volume = ed_volume.transpose(2, 0, 1)
        es_volume = es_volume.transpose(2, 0, 1)

        ed_volume = self.apply_volume_augmentations(ed_volume)
        es_volume = self.apply_volume_augmentations(es_volume)

        ed_volume = self.apply_volume_normalization(ed_volume, volume)
        es_volume = self.apply_volume_normalization(es_volume, volume)

        # We have to stack volume as batch
        ed_volume = np.expand_dims(ed_volume, axis=1)
        es_volume = np.expand_dims(es_volume, axis=1)

        ed_volume = torch.from_numpy(ed_volume)
        es_volume = torch.from_numpy(es_volume)

        if self.add_depth:
            ed_volume = data_utils.add_volume_depth_channels(ed_volume)
            es_volume = data_utils.add_volume_depth_channels(es_volume)

        return [ed_volume, es_volume, affine, header, initial_shape, str(external_code), original_ed, original_es]


def find_values(string, label, label_type):
    """

    Args:
        string:
        label:
        label_type:

    Returns:

    Example:
        string = "mms_centre14_vendorA"
        label = "centre"
        label_type = int
        -> res = [1, 4]

        string = "mms_centre14_vendorA"
        label = "vendor"
        label_type = str
        -> res = ['A']

    """
    res = None
    if string.find(label) != -1:
        c_centre = string[string.find(label) + len(label):]
        centre_break = c_centre.find("_")
        if centre_break != -1:
            c_centre = c_centre[:centre_break]
        res = [label_type(i) for i in c_centre]
    return res


def coral_dataset_selector(train_aug, train_aug_img, partition, args):
    vendor_loaders = []
    for coral_vendor in args.coral_vendors:
        vendor_loader = get_volume_loader(
            coral_vendor, train_aug, train_aug_img, add_depth=args.add_depth, partition=partition
        )
        vendor_loaders.append(vendor_loader)
    return vendor_loaders


class ACDC172Dataset(Dataset):
    """
    2D Dataset for ACDC Challenge.
    https://acdc.creatis.insa-lyon.fr/
    """

    def __init__(self, mode, transform, img_transform, add_depth=True, normalization="normalize", relative_path="",
                 train_patients=100):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        if mode not in ["train", "full_train", "validation"]:
            assert False, "Unknown mode '{}'".format(mode)

        self.base_dir = os.path.join(relative_path, "data/AC17")
        self.img_channels = 3 if add_depth else 1
        # MMS -> class_to_cat = {1: "LV", 2: "MYO", 3: "RV", 4: "Mean"}
        # Original ACDC -> class_to_cat = {1: "RV", 2: "MYO", 3: "LV", 4: "Mean"}
        self.class_to_cat = {1: "LV", 2: "MYO", 3: "RV", 4: "Mean"}
        self.map_classes = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}
        self.num_classes = 4
        self.include_background = False

        data = []
        for subdir, dirs, files in os.walk(self.base_dir):
            for file in files:
                entry = os.path.join(subdir, file)
                if "_gt" in entry and not ".nii" in entry and not ".nii.gz" in entry:
                    data.append(entry)

        if len(data) == 0:
            assert False, 'You have to transform volumes to 2D slices: ' \
                          'python tools/nifti2slices.py --data_path "data/AC17"'

        if mode in ["train", "validation"]:
            np.random.seed(1)
            patient_list = np.sort(np.unique([elem.split("/")[-2] for elem in data]))
            train_indx = np.random.choice(range(patient_list.shape[0]), size=(train_patients,), replace=False)
            ind = np.zeros(patient_list.shape[0], dtype=bool)
            ind[train_indx] = True
            val_indx = ~ind

            if mode == "train":
                data = [elem for elem in data if elem.split("/")[-2] in patient_list[train_indx]]
            elif mode == "validation":
                if train_patients > 85:
                    # If there are not too much patients take randomly
                    np.random.seed(1)
                    np.random.shuffle(data)
                    data = data[int(len(data) * .85):]
                else:
                    # Only get first 15 patients for validation, not ALL
                    data = [elem for elem in data if elem.split("/")[-2] in patient_list[val_indx][:15]]

        self.data = data
        self.mode = mode
        self.normalization = normalization

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)
        self.add_depth = add_depth

    def __len__(self):
        return len(self.data)

    @staticmethod
    def custom_collate(batch):
        """

        Args:
            batch: list of dataset items (from __getitem__). In this case batch is a list of dicts with
                   key image, and depending of validation or train different keys

        Returns:

        """
        # We have to modify "original_mask" as has different shapes
        batch_keys = list(batch[0].keys())
        res = {bkey: [] for bkey in batch_keys}
        for belement in batch:
            for bkey in batch_keys:
                res[bkey].append(belement[bkey])

        for bkey in batch_keys:
            if bkey == "original_mask" or bkey == "original_img" or bkey == "img_id":
                continue  # We wont stack over original_mask...
            res[bkey] = torch.stack(res[bkey])

        return res

    def __getitem__(self, idx):

        img_path = self.data[idx].replace("_gt", "")
        image = np.load(img_path)

        mask_path = self.data[idx]
        mask = np.load(mask_path)
        if self.map_classes:
            mask = map_mask_classes(mask, self.map_classes)

        img_id = os.path.splitext(img_path)[0].split("/")[-1]

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        image, mask = data_utils.apply_augmentations(image, self.transform, self.img_transform, mask)
        image = data_utils.apply_normalization(image, self.normalization)
        image = torch.from_numpy(np.expand_dims(image, axis=0))

        if self.add_depth:
            image = data_utils.add_depth_channels(image)
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float()

        return {
            "image": image, "original_img": original_image,
            "original_mask": original_mask, "label": mask, "img_id": img_id
        }


def dataset_selector(train_aug, train_aug_img, val_aug, args, is_test=False):
    train_datasets, val_datasets = [], []
    if "mms2d" in args.dataset:

        if is_test:
            test_dataset = MMsSubmissionDataset(
                transform=val_aug, img_transform=[],
                normalization=args.normalization, add_depth=args.add_depth,
            )

            return DataLoader(
                test_dataset, batch_size=1, shuffle=False, pin_memory=True,
                drop_last=False, collate_fn=test_dataset.simple_collate
            )

        only_end = False if "full" in args.dataset else True
        unlabeled = True if "unlabeled" in args.dataset else False
        c_centre, c_vendor = find_values(args.dataset, "centre", int), find_values(args.dataset, "vendor", str)

        if "all" in args.dataset:
            train_dataset = MMs2DDataset(
                partition="All", transform=train_aug, img_transform=train_aug_img,
                normalization=args.normalization,
                add_depth=args.add_depth, is_labeled=(not unlabeled), centre=c_centre, vendor=c_vendor,
                end_volumes=only_end, rand_histogram_matching=args.rand_histogram_matching
            )

            val_dataset = MMs2DDataset(
                partition="Validation", transform=val_aug, img_transform=[], normalization=args.normalization,
                add_depth=args.add_depth, is_labeled=False, centre=None, vendor=None, end_volumes=True
            )
        else:
            train_dataset = MMs2DDataset(
                partition="Training", transform=train_aug, img_transform=train_aug_img,
                normalization=args.normalization,
                add_depth=args.add_depth, is_labeled=(not unlabeled), centre=c_centre, vendor=c_vendor,
                end_volumes=only_end, rand_histogram_matching=args.rand_histogram_matching
                , patients_percentage=args.patients_percentage
            )

            val_dataset = MMs2DDataset(
                partition="Validation", transform=val_aug, img_transform=[], normalization=args.normalization,
                add_depth=args.add_depth, is_labeled=False, centre=None, vendor=None, end_volumes=True
            )

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    if "ACDC172D" in args.dataset:
        if is_test:
            assert False, "Not test partition available"
        train_dataset = ACDC172Dataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            add_depth=args.add_depth, normalization=args.normalization
        )

        val_dataset = ACDC172Dataset(
            mode="validation", transform=val_aug, img_transform=[],
            add_depth=args.add_depth, normalization=args.normalization
        )

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    if len(train_datasets) == 0 and len(train_datasets) == 0:
        assert False, f"Unknown dataset '{args.dataset}'"

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, pin_memory=True,
        shuffle=True, collate_fn=train_datasets[0].custom_collate
    )
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
        drop_last=False, collate_fn=val_datasets[0].custom_collate
    )

    num_classes, class_to_cat, include_background = [], None, None
    for dataset in train_datasets:
        num_classes.append(dataset.num_classes)
        if class_to_cat is None:
            class_to_cat = dataset.class_to_cat
            include_background = dataset.include_background
        else:
            if class_to_cat != dataset.class_to_cat:
                assert False, "Class to category from different datasets must be the same!"
            if include_background != dataset.include_background:
                assert False, "If one dataset includes background as a class, all should!"

    num_classes = np.array(num_classes)
    if not np.all(num_classes == num_classes[0]):
        print(f"Hay datasets con diferentes número de clases: {num_classes}")

    print(f"Train dataset len:  {len(train_dataset)}")
    print(f"Validation dataset len:  {len(val_dataset)}")
    return train_loader, val_loader, num_classes.max(), class_to_cat, include_background
