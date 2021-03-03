#!/usr/bin/env python
# coding: utf-8

# ---- My utils ----
from models import model_selector
from utils.dasegan_arguments import *
from utils.data_augmentation import data_augmentation_selector
from utils.datasets import dataset_selector
from utils.neural import *
from utils.gans import *

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
    input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm_layer,
    use_dropout=not args.no_dropout, gpu_ids=args.gpu, checkpoint=args.gen_checkpoint
)
discriminator = define_Dis(
    input_nc=3, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm_layer, gpu_ids=args.gpu,
    checkpoint=args.dis_checkpoint
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


print("\n --- START TRAINING ---")
set_grad([segmentator], False)
for epoch in range(args.epochs):

    current_generated_samples = 0
    lr = g_optimizer.param_groups[0]['lr']

    epoch_gen_loss, epoch_dis_loss = [], []

    for batch_indx, batch in enumerate(vol_loader):
        # step
        step = epoch * len(vol_loader) + batch_indx + 1
        vol_x = batch["image"].cuda()
        vol_x_original_label = torch.from_numpy(np.array(batch["vendor_label"])).cuda()
        img_id = batch["img_id"]

        #####################################################
        # -------------- Generator Computations -------------
        #####################################################

        set_grad([discriminator], False)
        g_optimizer.zero_grad()

        # -- Forward pass through generator --
        vol_u = generator(vol_x)

        pred_x = segmentator(vol_x)
        pred_u = segmentator(vol_u)

        # --- Identity/Cycle losses ---

        cycle_loss = L1(pred_x, pred_u) * args.cycle_coef

        # --- Adversarial losses ---
        vol_label_u = discriminator(vol_u)

        random_labels = get_random_labels(vol_x_original_label, AVAILABLE_LABELS)
        random_labels = labels2rfield(random_labels, vol_label_u.shape).to(vol_label_u.device)

        label_loss_u = MSE(vol_label_u, random_labels)

        # --- Total generators losses ---
        gen_loss = cycle_loss + label_loss_u
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
        vol_label_x = discriminator(vol_x)

        # --- Discriminator losses ---
        vol_x_original_label = labels2rfield(vol_x_original_label, vol_label_x.shape).to(vol_label_x.device)
        vol_x_label_dis_loss = MSE(vol_label_x, vol_x_original_label)

        # Total discriminators losses
        dis_loss = vol_x_label_dis_loss
        epoch_dis_loss.append(dis_loss.item())

        # --- Update discriminators ---
        dis_loss.backward()
        d_optimizer.step()

        if args.generated_samples > 0:
            for indx in range(len(vol_x)):
                if current_generated_samples < args.generated_samples:
                    c_pred_x = convert_multiclass_mask(pred_x[indx].unsqueeze(0)).data.cpu().numpy().squeeze()
                    c_pred_u = convert_multiclass_mask(pred_u[indx].unsqueeze(0)).data.cpu().numpy().squeeze()
                    plot_save_generated(
                        vol_x[indx].data.cpu().numpy()[0], vol_u[indx].data.cpu().numpy()[0], c_pred_x, c_pred_u,
                        os.path.join(args.output_dir, "generated_samples", f"epoch_{epoch}"), img_id[indx]
                    )
                    current_generated_samples += 1
                else:
                    break

    epoch_gen_loss, epoch_dis_loss = np.array(epoch_gen_loss).mean(), np.array(epoch_dis_loss).mean()
    print(f"Epoch {epoch} | lr: {lr} | Gen Loss: {epoch_gen_loss} | Dis Loss: {epoch_dis_loss}")

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
        f'{args.output_dir}/checkpoint_epoch{epoch+1}.pt'
    )

    # -- Update learning rates --
    g_lr_scheduler.step()
    d_lr_scheduler.step()
