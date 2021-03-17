#!/usr/bin/env python
# coding: utf-8

# ---- My utils ----
from models import model_selector
from models.gan import define_Gen, define_Dis
from utils.dasegan_arguments import *
from utils.data_augmentation import data_augmentation_selector
from utils.datasets import dataset_selector
from utils.neural import *
from utils.gans import *

import torch.nn.functional as F

set_seed(args.seed)

# Define Dataloaders
#####################################################
train_aug, train_aug_img, val_aug = data_augmentation_selector(
    args.data_augmentation, args.img_size, args.crop_size, args.mask_reshape_method
)
vol_loader, val_vols, num_classes, class_to_cat, include_background = dataset_selector(
    train_aug, train_aug_img, val_aug, args
)
print(f"Number of classes: {num_classes}")
AVAILABLE_LABELS = np.arange(0, vol_loader.dataset.num_vendors).tolist()
print(f"Number of vendors: {AVAILABLE_LABELS}")

# Define the networks
#####################################################
generator = define_Gen(
    input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.gen_norm_layer,
    use_dropout=not args.no_dropout, gpu_ids=args.gpu, checkpoint=args.gen_checkpoint
)
discriminator = define_Dis(
    input_nc=3, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.dis_norm_layer, gpu_ids=args.gpu,
    checkpoint=args.dis_checkpoint, real_fake=(args.realfake_coef > 0)
)
segmentator = model_selector(
    "segmentation", args.seg_net, num_classes,
    in_channels=3 if args.add_depth else 1, devices=args.gpu, checkpoint=args.seg_checkpoint
)

# Define Loss criterion

MSE = nn.MSELoss()
L1 = nn.L1Loss()

# Define Optimizers
#####################################################
g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    g_optimizer, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step
)
d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    d_optimizer, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step
)

print("\n\n --- START TRAINING --\n")
segmentator.eval()
set_grad([segmentator], False)

for epoch in range(args.epochs):

    current_generated_samples = 0
    lr = g_optimizer.param_groups[0]['lr']

    epoch_gen_loss, epoch_dis_loss = [], []
    vol_x_vendor_acc, vol_u_vendor_acc = [], []
    vol_x_realfake_acc, vol_u_realfake_acc = [], []
    vol_x_iou, vol_u_iou = [], []

    for batch_indx, batch in enumerate(vol_loader):

        vol_x = batch["image"].cuda()
        # Como utilizamos datos que no tienen porque estar etiquetados, recibimos una lista de labels
        # donde puede haber o no (None). Ejemplo: [None, tensor, None, None]
        inestable_label = batch["inestable_label"]
        vol_x_original_label = torch.from_numpy(np.array(batch["vendor_label"])).cuda()
        img_id = batch["img_id"]

        generator.train()
        discriminator.train()

        #####################################################
        # -------------- Generator Computations -------------
        #####################################################

        set_grad([discriminator], False)
        g_optimizer.zero_grad()

        # -- Forward pass through generator --
        vol_u = generator(vol_x)

        pred_x = segmentator(vol_x)

        pred_u = segmentator(vol_u)
        if args.use_original_mask:
            for mask_index, mask in enumerate(inestable_label):
                if mask is not None:
                    pred_x[mask_index] = F.one_hot(mask.squeeze().to(torch.int64), 4).permute(2, 0, 1).cuda()

        # --- Identity/Cycle losses ---
        cycle_loss = L1(pred_x, pred_u) * args.cycle_coef

        # --- Adversarial losses: Vendor Label ---
        vol_fake_label_u, vol_vendor_label_u = discriminator(vol_u)

        random_labels = get_random_labels(vol_x_original_label, AVAILABLE_LABELS)
        random_labels = labels2rfield(random_labels, vol_vendor_label_u.shape).to(vol_vendor_label_u.device)

        vendor_label_loss_u = MSE(vol_vendor_label_u, random_labels)

        # --- Adversarial losses: Real/Fake Label ---
        fake_label_loss_u = 0
        if args.realfake_coef > 0:
            target_real = torch.ones_like(vol_fake_label_u).cuda()
            fake_label_loss_u = MSE(vol_fake_label_u, target_real) * args.realfake_coef

        # --- Total generators losses ---
        gen_loss = cycle_loss + vendor_label_loss_u + fake_label_loss_u
        epoch_gen_loss.append(gen_loss.item())

        #  --- Update generators ---
        gen_loss.backward()
        g_optimizer.step()

        #####################################################
        # ------------ Discriminator Computations -----------
        #####################################################

        set_grad([discriminator], True)
        d_optimizer.zero_grad()

        # --- Forward pass through discriminators ---
        vol_real_label_x, vol_label_x = discriminator(vol_x)
        vol_fake_label_u, vol_label_u = discriminator(vol_u.detach())

        # --- Discriminator losses ---
        vol_x_original_label = labels2rfield(vol_x_original_label, vol_label_x.shape).to(vol_label_x.device)
        vol_x_label_dis_loss = MSE(vol_label_x, vol_x_original_label) * 0.5
        vol_u_label_dis_loss = MSE(vol_label_u, vol_x_original_label) * args.dis_u_coef

        # -- Real/Fake Label --
        real_fake_loss = 0
        if args.realfake_coef > 0:
            target_real = torch.ones_like(vol_real_label_x).cuda()
            real_loss_x = MSE(vol_real_label_x, target_real)

            target_fake = torch.zeros_like(vol_fake_label_u).cuda()
            fake_loss_u = MSE(vol_fake_label_u, target_fake)

            real_fake_loss = (real_loss_x + fake_loss_u) * args.realfake_coef

        # Total discriminators losses
        dis_loss = vol_x_label_dis_loss + vol_u_label_dis_loss + real_fake_loss
        epoch_dis_loss.append(dis_loss.item())

        # --- Update discriminators ---
        dis_loss.backward()
        d_optimizer.step()

        #####################################################
        # ---------------- EVALUATION METRICS ---------------
        #####################################################

        set_grad([discriminator], False)
        discriminator.eval()

        # --- Discriminator metrics ---
        vol_real_label_x, vol_label_x = discriminator(vol_x)
        vol_fake_label_u, vol_label_u = discriminator(vol_u.detach())

        _, vol_label_x = torch.max(vol_label_x.data, 1)
        _, vol_label_u = torch.max(vol_label_u.data, 1)
        label_size = np.prod(list(vol_x_original_label.shape))
        vol_x_vendor_acc.append((torch.sum(vol_label_x == vol_x_original_label.squeeze()) / label_size).item())
        vol_u_vendor_acc.append(((torch.sum(vol_label_u == vol_x_original_label.squeeze()) / label_size).item()))

        if args.realfake_coef > 0:
            vol_x_realfake_acc.append(
                (torch.sum((vol_real_label_x > 0.5) == torch.ones_like(vol_real_label_x).cuda()) / label_size).item()
            )
            vol_u_realfake_acc.append(
                (torch.sum((vol_fake_label_u > 0.5) == torch.zeros_like(vol_real_label_x).cuda()) / label_size).item()
            )

        # --- Segmentator metrics ---
        pred_x = segmentator(vol_x)
        for mask_index, mask in enumerate(inestable_label):
            if mask is not None:
                vol_x_iou.append(evaluate_segmentation(pred_x.detach()[mask_index], mask.squeeze()))
                vol_u_iou.append(evaluate_segmentation(pred_u.detach()[mask_index], mask.squeeze()))

        # --- Plot examples ---
        if args.generated_samples > 0 and (current_generated_samples < args.generated_samples):

            for mask_index, mask in enumerate(inestable_label):
                if mask is not None:

                    if current_generated_samples < args.generated_samples:
                        c_pred_x = convert_multiclass_mask(pred_x[mask_index].unsqueeze(0)).data.cpu().numpy().squeeze()
                        c_pred_u = convert_multiclass_mask(pred_u[mask_index].unsqueeze(0)).data.cpu().numpy().squeeze()
                        plot_save_generated(
                            vol_x[mask_index].data.cpu().numpy()[0], mask.squeeze(), c_pred_x,
                            vol_u[mask_index].data.cpu().numpy()[0], c_pred_u,
                            os.path.join(args.output_dir, "generated_samples", f"epoch_{epoch}"), img_id[mask_index]
                        )
                        current_generated_samples += 1

                    else:
                        break

    vol_x_vendor_acc, vol_u_vendor_acc = np.array(vol_x_vendor_acc).mean(), np.array(vol_u_vendor_acc).mean()
    dis_metrics = f"X Vendor Acc: {vol_x_vendor_acc:.4f} | U Vendor Acc: {vol_u_vendor_acc:.4f}"
    if args.realfake_coef > 0:
        vol_x_realfake_acc = np.array(vol_x_realfake_acc).mean()
        vol_u_realfake_acc = np.array(vol_u_realfake_acc).mean()
        dis_metrics += f" | X RealFake Acc: {vol_x_realfake_acc:.4f} | U RealFake Acc: {vol_u_realfake_acc:.4f}"

    seg_metrics = f" Vol X IOU: {np.array(vol_x_iou).mean():.4f} | Vol U IOU: {np.array(vol_u_iou).mean():.4f}"

    epoch_gen_loss, epoch_dis_loss = np.array(epoch_gen_loss).mean(), np.array(epoch_dis_loss).mean()
    print(
        f"Epoch {epoch} | lr: {lr} | Gen Loss: {epoch_gen_loss:.4f} | Dis Loss: {epoch_dis_loss:.4f} "
        f"| {dis_metrics} | {seg_metrics}"
    )

    # ---- Checkpoint ----
    torch.save(
        {
            'epoch': epoch + 1,
            'discriminator': discriminator.state_dict(),
            'segmentator': segmentator.state_dict(),
            'generator': generator.state_dict(),
            'd_optimizer': d_optimizer.state_dict(),
            'g_optimizer': g_optimizer.state_dict()
        },
        f'{args.output_dir}/checkpoint_epoch{epoch + 1}.pt'
    )

    # -- Update learning rates --
    g_lr_scheduler.step()
    d_lr_scheduler.step()
