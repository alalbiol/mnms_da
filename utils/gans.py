import os
import matplotlib.pyplot as plt
import copy
import numpy as np

import torch


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


def labels2rfield(labels, shape):
    # vol_label_u has shape [batch, channels, receptive_field, receptive_field], to be able to multiply
    # with random labels, we have to transform list labels shape [batch] to [batch, 1, 1, 1]
    # eg. labels -> [0,1,0,2,0]
    labels = labels.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    labels = torch.ones(shape).to(labels.device) * labels
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

    ax1[1].imshow(original_img_mask, cmap="gray")
    ax1[1].set_title("Original - Mask")

    ax2[0].imshow(generated_img, cmap="gray")
    ax2[0].set_title("Generated Image")

    ax2[1].imshow(generated_img_mask, cmap="gray")
    ax2[1].set_title("Generated - Mask")

    ax3[0].imshow(original_img, cmap="gray")
    ax3[0].set_title("Original Image")

    ax3[1].imshow(original_img_pred_mask, cmap="gray")
    ax3[1].set_title("Predicted - Mask")

    pred_filename = os.path.join(
        save_dir,
        f"generated_{img_id}.png",
    )
    plt.savefig(pred_filename, dpi=200, bbox_inches='tight')
    plt.close()


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
