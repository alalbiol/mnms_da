import os
import matplotlib.pyplot as plt
import copy
import numpy as np
import matplotlib.gridspec as gridspec

import torch

from utils.datasets import get_mnms_arrays


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class SampleFromPool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


def get_random_labels(vol_x_original_label, available_labels):
    """
    vol_x_original_label -> tensor([0,1,1,2,3,0])
    available_labels -> [0,1,2] (num available vendors, ej. A-B-C)
    returns -> Different random labels tensor([1,2,0,1,1,1]) within available_labels
    """
    res = []
    for l in vol_x_original_label:
        res.append(
            np.random.choice([x for x in available_labels if x != l], 1)[0]
        )
    return torch.from_numpy(np.array(res))


def labels2rfield(method, shape, label_range=None, labels=None):
    # vol_label_u has shape [batch, channels, receptive_field, receptive_field], to be able to multiply
    # with random labels, we have to transform list labels shape [batch] to [batch, 1, 1]
    # eg. labels -> [0,1,0,2,0]

    if method == "random_maps" or method == "maps":
        batch, channels, height, width = shape
        labels = labels.unsqueeze(1).unsqueeze(1)
        labels = torch.ones((batch, height, width), dtype=torch.long) * labels
    elif method == "random_atomic":
        batch, channels, height, width = shape
        min_val, max_val = label_range
        labels = torch.randint(min_val, max_val, (batch, height, width), dtype=torch.long)
    else:
        assert False, f"Unknown labels2rfield method '{method}'"
    return labels


def plot_save_generated(
        original_img, original_img_mask, original_img_pred_mask, generated_img, generated_img_mask,
        save_dir, img_id
):

    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.005, hspace=0.2, right=0.5)

    ax1[0].axis('off')
    ax1[1].axis('off')
    ax2[0].axis('off')
    ax2[1].axis('off')
    ax3[0].axis('off')
    ax3[1].axis('off')

    ax1[0].imshow(original_img, cmap="gray")
    ax1[0].set_title("Original Image")

    ax1[1].imshow(original_img_mask, cmap="gray", vmin=0, vmax=3)
    ax1[1].set_title("Original - Mask")

    ax2[0].imshow(generated_img, cmap="gray")
    ax2[0].set_title("Generated Image")

    ax2[1].imshow(generated_img_mask, cmap="gray", vmin=0, vmax=3)
    ax2[1].set_title("Generated - Mask")

    ax3[0].imshow(original_img, cmap="gray")
    ax3[0].set_title("Original Image")

    ax3[1].imshow(original_img_pred_mask, cmap="gray", vmin=0, vmax=3)
    ax3[1].set_title("Predicted - Mask")

    pred_filename = os.path.join(
        save_dir,
        f"generated_{img_id}.png",
    )
    plt.savefig(pred_filename, dpi=200, bbox_inches='tight')
    plt.close()


def plot_save_generated_vendor_list(
        transformed_samples, pred_path, rows=10, cols=4
):
    a_transformed, b_transformed, c_transformed, d_transformed = transformed_samples
    fig = plt.figure(figsize=(cols + 1, rows + 1))

    gs1 = gridspec.GridSpec(rows, cols, )
    gs1.update(wspace=0.02, hspace=0.05)  # set the spacing between axes.
    set_title = {"A": True, "B": True, "C": True, "D": True}

    curr_vendor = 0  # 0 -> A, 1 -> B, 2 -> C, 3 -> D
    for i in range(rows*cols):
        # i = i + 1 # grid spec indexes from 0
        ax = plt.subplot(gs1[i])
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('auto')

        if curr_vendor == 0:
            if set_title["A"]:
                ax.set_title("Vendor A")
                set_title["A"] = False
            img = a_transformed[i]
        elif curr_vendor == 1:
            if set_title["B"]:
                ax.set_title("Vendor B")
                set_title["B"] = False
            img = b_transformed[i]
        elif curr_vendor == 2:
            if set_title["C"]:
                ax.set_title("Vendor C")
                set_title["C"] = False
            img = c_transformed[i]
        elif curr_vendor == 3:
            if set_title["D"]:
                ax.set_title("Vendor D")
                set_title["D"] = False
            img = d_transformed[i]
        else:
            assert False, "Unknown plotting error. Current vendor > 3?!"
        ax.imshow(img, cmap="gray")

        curr_vendor += 1
        curr_vendor = curr_vendor if curr_vendor <= 3 else 0

    fig.savefig(pred_path, dpi=200, bbox_inches='tight')
    plt.close()
    return fig


def plot_save_kernels(kernel, save_dir,  path_info="", by_kernel=True):
    """
    kernel with shape: (batch, channels, height, width)
    """

    os.makedirs(save_dir, exist_ok=True)

    height, width = kernel.shape[2], kernel.shape[3]

    for sample in range(kernel.shape[0]):

        if by_kernel:

            for kernel_index in range(kernel.shape[1]):

                plt.figure(figsize=(14, 6))
                plt.axis('off')
                plt.imshow(kernel[sample, kernel_index, ...], cmap="gray")

                kernel_save_dir = os.path.join(
                    save_dir, f"sample_{sample}", f"{height}x{width}{path_info}"
                )
                os.makedirs(kernel_save_dir, exist_ok=True)
                pred_filename = os.path.join(
                    kernel_save_dir,
                    f"kernel_{kernel_index}.png",
                )
                plt.savefig(pred_filename, dpi=200, bbox_inches='tight')
                plt.close()

        else:

            plt.figure(figsize=(14, 6))
            plt.axis('off')
            plt.imshow(kernel[sample, ...], cmap="gray")

            kernel_save_dir = os.path.join(
                save_dir, f"sample_{sample}", f"{height}x{width}{path_info}"
            )
            os.makedirs(kernel_save_dir, exist_ok=True)
            pred_filename = os.path.join(
                kernel_save_dir,
                f"{kernel.shape[1]}kernels.png",
            )
            plt.savefig(pred_filename, dpi=200, bbox_inches='tight')
            plt.close()


def get_vendors_samples(normalization):
    rows, cols = 10, 4
    num_test_samples = rows * cols

    img_list_a = get_mnms_arrays(
        "A", normalization=normalization, add_depth=True,
    )
    img_list_b = get_mnms_arrays(
        "B", normalization=normalization, add_depth=True
    )
    img_list_c = get_mnms_arrays(
        "C", normalization=normalization, data_mod="_unlabeled", add_depth=True
    )
    img_list_d = get_mnms_arrays(
        "D", normalization=normalization, data_mod="_unlabeled_full", partition="Testing", add_depth=True
    )

    np.random.seed(42)
    img_list_a_selection = img_list_a[np.random.choice(len(img_list_a), num_test_samples, replace=False)]
    np.random.seed(42)
    img_list_b_selection = img_list_b[np.random.choice(len(img_list_b), num_test_samples, replace=False)]
    np.random.seed(42)
    img_list_c_selection = img_list_c[np.random.choice(len(img_list_c), num_test_samples, replace=False)]
    np.random.seed(42)
    img_list_d_selection = img_list_d[np.random.choice(len(img_list_d), num_test_samples, replace=False)]

    return [img_list_a_selection, img_list_b_selection, img_list_c_selection, img_list_d_selection]
