import os
import matplotlib.pyplot as plt
from time import gmtime, strftime
import math
import numpy as np
import pandas as pd
import cv2


def current_time():
    """
    Gives current time
    :return: (String) Current time formated %Y-%m-%d %H:%M:%S
    """
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())


def dict2df(my_dict, path):
    """
    Save python dictionary as csv using pandas dataframe
    :param my_dict: Dictionary like {"epoch": [1, 2], "accuracy": [0.5, 0.9]}
    :param path: /path/to/file.csv
    :return: (void) Save csv on specified path
    """
    df = pd.DataFrame.from_dict(my_dict, orient="columns")
    df.index.names = ['epoch']
    df.to_csv(path, index=True)


def convert_multiclass_mask(mask):
    """
    Transform multiclass mask [batch, num_classes, h, w] to [batch, h, w]
    :param mask: Mask to transform
    :return: Transformed multiclass mask
    """
    return mask.max(1)[1]


def map2multiclass(mask):
    """
    Transform multiclass mask [batch, num_classes, h, w] to [batch, h, w]
    :param mask: Mask to transform
    :return: Transformed multiclass mask
    """
    return mask.max(1)[1]


def map_mask_classes(mask, classes_map):
    """

    Args:
        mask: (np.array) Mask Array to map (height, width)
        classes_map: (dict) Mapping between classes. E.g.  {0:0, 1:3, 2:2, 3:1 ,4:4}

    Returns: (np.array) Mapped mask array

    """
    res = np.zeros_like(mask).astype(mask.dtype)
    for value in np.unique(mask):
        if value not in classes_map:
            assert False, f"Please specify all class maps. {value} not in {classes_map}"
        res += np.where(mask == value, classes_map[value], 0).astype(mask.dtype)
    return res


def reshape_masks(ndarray, to_shape, mask_reshape_method):
    """

    Args:
        ndarray: (np.array) Mask Array to reshape
        to_shape: (tuple) Final desired shape
        mask_reshape_method:

    Returns: (np.array) Reshaped array to desired shape

    """

    h_in, w_in = ndarray.shape
    h_out, w_out = to_shape

    if mask_reshape_method == "padd":

        if h_in > h_out:  # center crop along h dimension
            h_offset = math.ceil((h_in - h_out) / 2)
            ndarray = ndarray[h_offset:(h_offset + h_out), :]
        else:  # zero pad along h dimension
            pad_h = (h_out - h_in)
            rem = pad_h % 2
            pad_dim_h = (math.ceil(pad_h / 2), math.ceil(pad_h / 2 + rem))
            # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
            npad = (pad_dim_h, (0, 0))
            ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)

        if w_in > w_out:  # center crop along w dimension
            w_offset = math.ceil((w_in - w_out) / 2)
            ndarray = ndarray[:, w_offset:(w_offset + w_out)]
        else:  # zero pad along w dimension
            pad_w = (w_out - w_in)
            rem = pad_w % 2
            pad_dim_w = (math.ceil(pad_w / 2), math.ceil(pad_w / 2 + rem))
            npad = ((0, 0), pad_dim_w)
            ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)

    elif mask_reshape_method == "resize":
        ndarray = cv2.resize(ndarray.astype('float32'), (w_out, h_out))
    else:
        assert False, f"Unknown mask resize method '{mask_reshape_method}'"

    return ndarray  # reshaped


def binarize_volume_prediction(volume_pred, original_shape=None, mask_reshape_method="padd"):
    """
    Takes a prediction mask with shape [slices, classes, height, width]
    and binarizes it to [slices, height, width]
    :param volume_pred: (array) [slices, classes, height, width] volume mask predictions
    :param original_shape: (tuple) Original volume shape to reshape to
    :param mask_reshape_method:
    :return: (array) [slices, height, width] volume binarized mask
    """
    s, c, h, w = volume_pred.shape
    if original_shape is not None: h, w, s = original_shape
    output_volume = np.empty([s, h, w])

    for indx, single_pred in enumerate(volume_pred):
        pred_mask = convert_multiclass_mask(single_pred.unsqueeze(0)).data.cpu().numpy()

        if original_shape is not None:
            # Resize prediction to original image shape
            pred_mask = reshape_masks(pred_mask.squeeze(0), (h, w), mask_reshape_method)
        output_volume[indx, ...] = pred_mask

    return output_volume


def plot_save_pred(original_img, original_mask, pred_mask, save_dir, img_id):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    ax1.imshow(original_img, cmap="gray")
    ax1.set_title("Original Image")

    masked = np.ma.masked_where(original_mask == 0, original_mask)
    ax2.imshow(original_img, cmap="gray")
    ax2.imshow(masked, 'jet', interpolation='bilinear', alpha=0.25)
    ax2.set_title("Original Overlay")

    masked = np.ma.masked_where(pred_mask == 0, pred_mask)
    ax3.imshow(original_img, cmap="gray")
    ax3.imshow(masked, 'jet', interpolation='bilinear', alpha=0.25)
    ax3.set_title("Prediction Overlay")

    pred_filename = os.path.join(
        save_dir,
        f"mask_pred_{img_id}.png",
    )
    plt.savefig(pred_filename, dpi=200, pad_inches=0.2, bbox_inches='tight')
    plt.close()


def plot_save_pred_volume(img_volume, pred_mask_volume, save_dir, img_id):
    """
    Save overlays of images and predictions using volumes
    :param img_volume: (array) [height, width, slices] Original image
    :param pred_mask_volume: (array) [height, width, slices] Prediction mask
    :param save_dir: (string) Folder to save overlays
    :param img_id: (string) Image identifier
    :return:
    """

    os.makedirs(save_dir, exist_ok=True)

    img_volume = img_volume.transpose(2, 0, 1)  # [height, width, slices] -> [slices, height, width]
    pred_mask_volume = pred_mask_volume.transpose(2, 0, 1)  # [height, width, slices] -> [slices, height, width]

    for indx, (img) in enumerate(img_volume):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))
        ax1.axis('off')
        ax2.axis('off')

        ax1.imshow(img, cmap="gray")
        ax1.set_title("Original Image")

        mask = pred_mask_volume[indx, ...]
        masked = np.ma.masked_where(mask == 0, mask)
        ax2.imshow(img, cmap="gray")
        ax2.imshow(masked, 'jet', interpolation='bilinear', alpha=0.25)
        ax2.set_title("Original Overlay")

        pred_filename = os.path.join(
            save_dir,
            "mask_pred_{}_slice{}.png".format(img_id, indx),
        )
        plt.savefig(pred_filename, dpi=200, pad_inches=0.2, bbox_inches='tight')
        plt.close()


def linear_rampup(total_epochs, current_epoch, final_value):
    return (current_epoch / total_epochs) * final_value
