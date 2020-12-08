from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from skimage import io
import albumentations
import copy
import pandas as pd

import utils.dataload as data_utils
from tools.metrics_mnms import load_nii


class MMs2DDataset(Dataset):
    """
    Dataset for Digital Retinal Images for Vessel Extraction (DRIVE) Challenge.
    https://drive.grand-challenge.org/
    """

    def __init__(self, partition, transform, img_transform, normalization="normalize", add_depth=True,
                 is_labeled=True, centre=None, vendor=None, end_volumes=True, data_relative_path=""):
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
        """

        if partition not in ["Training", "Validation", "Testing"]:
            assert False, "Unknown mode '{}'".format(partition)

        self.base_dir = os.path.join(data_relative_path, "data/MMs")
        self.partition = partition
        self.img_channels = 3
        self.class_to_cat = {1: "LV", 2: "MYO", 3: "RV", 4: "Mean"}
        self.include_background = False
        self.num_classes = 4  # background - LV - MYO - RV

        data = pd.read_csv(os.path.join(self.base_dir, "slices_info.csv"))
        data = data.loc[(data["Partition"] == partition) & (data["Labeled"] == is_labeled)]
        if vendor is not None:
            data = data.loc[data['Vendor'].isin(vendor)]
        if centre is not None:
            data = data.loc[data['Centre'].isin(centre)]

        if end_volumes:  # Get only volumes in 'ED' and 'ES' phases (segmented)
            data = data.loc[(data["ED"] == data["Phase"]) | (data["ES"] == data["Phase"])]

        data = data.reset_index(drop=True)
        self.data = data

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

    def __getitem__(self, idx):
        df_entry = self.data.loc[idx]
        external_code = df_entry["External code"]
        c_slice = df_entry["Slice"]
        c_phase = df_entry["Phase"]
        c_vendor = df_entry["Vendor"]
        c_centre = df_entry["Centre"]
        img_id = f"{external_code}_slice{c_slice}_phase{c_phase}_vendor{c_vendor}_centre{c_centre}"

        labeled_info = ""
        if self.partition == "Training":
            labeled_info = "Labeled" if df_entry["Labeled"] else "Unlabeled"

        img_path = os.path.join(
            self.base_dir, self.partition, labeled_info, external_code,
            f"{external_code}_sa_slice{c_slice}_phase{c_phase}.npy"
        )
        image = np.load(img_path)

        mask = None
        if not (self.partition == "Training" and not df_entry["Labeled"]):
            mask_path = os.path.join(
                self.base_dir, self.partition, labeled_info, external_code,
                f"{external_code}_sa_gt_slice{c_slice}_phase{c_phase}.npy"
            )
            mask = np.load(mask_path)

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        image, mask = data_utils.apply_augmentations(image, self.transform, self.img_transform, mask)
        image = data_utils.apply_normalization(image, self.normalization)
        image = torch.from_numpy(np.expand_dims(image, axis=0))

        if self.add_depth:
            image = data_utils.add_depth_channels(image)
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float() if mask is not None else None

        return {
            "img_id": img_id, "image": image, "label": mask,
            "original_img": original_image, "original_mask": original_mask
        }


class MMsSubmissionDataset(Dataset):
    """
    Submission Dataset for Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms).
    """

    def __init__(self, transform, img_transform, add_depth=False, normalization="standardize",
                 data_relative_path=""):
        """
        :param input_dir: (string) Path with volume folders (usually where info.csv is located)
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param add_depth: (bool) If apply image transformation 1 to 3 channels or not
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        self.img_channels = 3
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

    def apply_volume_normalization(self, list_images):
        for indx, image in enumerate(list_images):
            list_images[indx, ...] = data_utils.apply_normalization(image, self.normalization)
        return list_images

    def add_volume_depth_channels(self, list_images):
        b, d, h, w = list_images.shape
        new_list_images = torch.empty((b, 3, h, w))
        for indx, image in enumerate(list_images):
            new_list_images[indx, ...] = data_utils.add_depth_channels(image)
        return new_list_images

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

        ed_volume = self.apply_volume_normalization(ed_volume)
        es_volume = self.apply_volume_normalization(es_volume)

        # We have to stack volume as batch
        ed_volume = np.expand_dims(ed_volume, axis=1)
        es_volume = np.expand_dims(es_volume, axis=1)

        ed_volume = torch.from_numpy(ed_volume)
        es_volume = torch.from_numpy(es_volume)

        if self.add_depth:
            ed_volume = self.add_volume_depth_channels(ed_volume)
            es_volume = self.add_volume_depth_channels(es_volume)

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


def dataset_selector(train_aug, train_aug_img, val_aug, args, is_test=False):
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

        train_dataset = MMs2DDataset(
            partition="Training", transform=train_aug, img_transform=train_aug_img, normalization=args.normalization,
            add_depth=args.add_depth, is_labeled=(not unlabeled), centre=c_centre, vendor=c_vendor, end_volumes=only_end
        )

        val_dataset = MMs2DDataset(
            partition="Validation", transform=val_aug, img_transform=[], normalization=args.normalization,
            add_depth=args.add_depth, is_labeled=False, centre=None, vendor=None, end_volumes=True
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, pin_memory=True,
            shuffle=True, collate_fn=train_dataset.custom_collate
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
            drop_last=False, collate_fn=val_dataset.custom_collate
        )

    else:
        assert False, f"Unknown dataset '{args.dataset}'"

    print(f"Train dataset len:  {len(train_dataset)}")
    print(f"Validation dataset len:  {len(val_dataset)}")
    return train_loader, val_loader
