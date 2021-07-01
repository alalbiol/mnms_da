#!/usr/bin/env python
# coding: utf-8
import wandb

# ---- My utils ----
from models import model_selector
from models.gan import define_Gen, define_Dis
from utils.dasegan_arguments import *
from utils.data_augmentation import data_augmentation_selector
from utils.datasets import dataset_selector
from utils.logging import get_name
from utils.neural import *
from utils.gans import *

os.environ["WANDB_SILENT"] = "true"
set_seed(args.seed)

# Define Dataloader
#####################################################
train_aug, train_aug_img, val_aug = data_augmentation_selector(
    args.data_augmentation, args.img_size, args.crop_size, args.mask_reshape_method
)
vol_loader, val_vols, num_classes, class_to_cat, include_background = dataset_selector(
    train_aug, train_aug_img, val_aug, args, sampler="weighted_sampler"
)

print(f"Number of segmentator classes: {num_classes}")
AVAILABLE_LABELS = list(np.arange(0, vol_loader.dataset.num_vendors))
print(f"Number of vendors: {AVAILABLE_LABELS}")

# Define the networks
#####################################################
generator = define_Gen(
    input_nc=3 if args.add_depth else 1, output_nc=3 if args.add_depth else 1, ngf=args.ngf, netG=args.gen_net,
    norm=args.gen_norm_layer, use_dropout=not args.no_dropout, gpu_ids=args.gpu, checkpoint=args.gen_checkpoint
)
discriminator = define_Dis(
    input_nc=3 if args.add_depth else 1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.dis_norm_layer,
    gpu_ids=args.gpu, checkpoint=args.dis_checkpoint, real_fake=(args.realfake_coef > 0),
    num_classes=len(AVAILABLE_LABELS)
)
segmentator = model_selector(
    "segmentation", args.seg_net, num_classes,
    in_channels=3 if args.add_depth else 1, devices=args.gpu, checkpoint=args.seg_checkpoint
)

# Define Loss criterion
dis_labels_criterion = get_loss(args.dis_labels_criterion)
dis_realfake_criterion = get_loss(args.dis_realfake_criterion)
identity_mask_criterion = nn.L1Loss()
task_criterion, task_weights_criterion, task_multiclass_criterion = get_criterion(
    args.task_criterion, args.task_weights_criterion
)

# Define Optimizers
#####################################################
g_optimizer = torch.optim.Adam([
    {"params": segmentator.parameters(), "lr": args.segmentator_lr},
    {"params": generator.parameters(), "lr": args.generator_lr},
], betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(0.5, 0.999))

g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    g_optimizer, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step
)
d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    d_optimizer, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step
)

vendors_samples = None
if args.plot_examples:
    print("Generating samples to plot...")
    vendors_samples = get_vendors_samples(args.normalization, add_depth=args.add_depth, num_test_samples=10)

wandb.init(project="MnMs DASEGAN - Exp2", name=get_name(args.unique_id), config=args)  # name="experiment1",

print("\n\n --- START TRAINING --\n")

# wandb.watch(generator)
# wandb.watch(discriminator)

for epoch in range(args.epochs):

    epoch_gen_loss, epoch_dis_loss = [], []

    epoch_dis_realfake_loss, epoch_dis_xlabel_loss, epoch_dis_ulabel_loss = [], [], []
    epoch_gen_cycle_loss, epoch_gen_ulabel_loss, epoch_gen_ufake_loss = [], [], []
    epoch_seg_xtask_loss, epoch_seg_utask_loss = [], []

    vol_x_vendor_acc, vol_u_vendor_acc = [], []
    vol_x_realfake_acc, vol_u_realfake_acc = [], []
    vol_x_iou, vol_u_iou = [], []

    for batch_indx, batch in enumerate(vol_loader):

        vol_x = batch["image"].cuda()
        # Como utilizamos datos que no tienen porque estar etiquetados, recibimos una lista de labels
        # donde puede haber o no (None). Ejemplo: [None, tensor, None, None]
        inestable_mask = batch["inestable_mask"]
        vol_x_original_label = torch.from_numpy(np.array(batch["vendor_label"])).cuda()
        img_id = batch["img_id"]

        generator.train()
        discriminator.train()
        segmentator.train()

        ###################################################################
        # -------------- Generator & Segmentator Computations -------------
        ###################################################################

        set_grad([discriminator], False)
        set_grad([segmentator, generator], True)
        g_optimizer.zero_grad()

        # -- Forward pass through generator --
        vol_u = generator(vol_x)

        pred_x = segmentator(vol_x)
        pred_u = segmentator(vol_u)

        # --- Task Loss ---
        if not all(m is None for m in inestable_mask):
            original_masks = torch.vstack([imask for imask in inestable_mask if imask is not None]).to(pred_x.device)
            masked_indices = torch.tensor([index for index, imask in enumerate(inestable_mask) if imask is not None])
            masked_indices = masked_indices.to(pred_x.device)

            task_loss_x = calculate_loss(
                original_masks, torch.index_select(pred_x, 0, masked_indices),
                task_criterion, task_weights_criterion, task_multiclass_criterion, num_classes
            )

            task_loss_u = calculate_loss(
                original_masks, torch.index_select(pred_u, 0, masked_indices),
                task_criterion, task_weights_criterion, task_multiclass_criterion, num_classes
            ) * linear_rampup(args.epochs, epoch + 1, args.task_loss_u_coef)

            task_loss = task_loss_x + task_loss_u
        else:
            task_loss = task_loss_x = task_loss_u = torch.tensor(0).to(pred_x.device)

        # --- Identity/Cycle losses ---
        cycle_loss = identity_mask_criterion(pred_x, pred_u) * args.cycle_coef

        # --- Adversarial losses: Vendor Label ---
        vol_fake_label_u, vol_vendor_label_u = discriminator(vol_u)

        random_labels = get_random_labels(vol_x_original_label, AVAILABLE_LABELS)
        random_labels = labels2rfield(
            method=args.rfield_method, shape=vol_vendor_label_u.shape,
            label_range=(0, len(AVAILABLE_LABELS) + 1), labels=random_labels
        ).to(vol_vendor_label_u.device)

        vendor_label_loss_u = dis_labels_criterion(vol_vendor_label_u, random_labels) * args.vendor_label_coef

        # --- Adversarial losses: Real/Fake Label ---
        fake_label_loss_u = 0
        if args.realfake_coef > 0:
            target_real = torch.ones_like(vol_fake_label_u).cuda()
            fake_label_loss_u = dis_realfake_criterion(vol_fake_label_u, target_real) * args.realfake_coef

        # --- Total generators losses ---
        gen_loss = cycle_loss + vendor_label_loss_u + fake_label_loss_u + task_loss
        epoch_gen_loss.append(gen_loss.item())
        epoch_gen_cycle_loss.append(cycle_loss.item())
        epoch_gen_ulabel_loss.append(vendor_label_loss_u.item())
        epoch_seg_xtask_loss.append(task_loss_x.item())
        epoch_seg_utask_loss.append(task_loss_u.item())
        epoch_gen_ufake_loss.append(0 if fake_label_loss_u == 0 else fake_label_loss_u.item())

        #  --- Update generators ---
        gen_loss.backward()
        g_optimizer.step()

        #####################################################
        # ------------ Discriminator Computations -----------
        #####################################################

        set_grad([discriminator], True)
        set_grad([segmentator, generator], False)
        d_optimizer.zero_grad()

        # --- Forward pass through discriminators ---
        vol_real_label_x, vol_label_x = discriminator(vol_x)
        vol_fake_label_u, vol_label_u = discriminator(vol_u.detach())

        # --- Discriminator losses ---
        vol_x_original_label_rfield = labels2rfield(
            method="maps", shape=vol_label_x.shape, labels=vol_x_original_label
        ).to(vol_label_x.device)
        vol_x_label_dis_loss = dis_labels_criterion(vol_label_x, vol_x_original_label_rfield)
        vol_u_label_dis_loss = dis_labels_criterion(vol_label_u, vol_x_original_label_rfield) * args.dis_u_coef

        # -- Real/Fake Label --
        real_fake_loss = 0
        if args.realfake_coef > 0:
            target_real = torch.ones_like(vol_real_label_x).cuda()
            real_loss_x = dis_realfake_criterion(vol_real_label_x, target_real)

            target_fake = torch.zeros_like(vol_fake_label_u).cuda()
            fake_loss_u = dis_realfake_criterion(vol_fake_label_u, target_fake)

            real_fake_loss = (real_loss_x + fake_loss_u) * args.realfake_coef

        # Total discriminators losses
        dis_loss = vol_x_label_dis_loss + vol_u_label_dis_loss + real_fake_loss
        epoch_dis_loss.append(dis_loss.item())
        epoch_dis_xlabel_loss.append(vol_x_label_dis_loss.item())
        epoch_dis_ulabel_loss.append(vol_u_label_dis_loss.item())
        epoch_dis_realfake_loss.append(0 if real_fake_loss == 0 else real_fake_loss.item())

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

        vol_x_original_label_rfield = labels2rfield(
            method="maps", shape=vol_label_x.shape, labels=vol_x_original_label
        ).to(vol_label_x.device)

        vol_label_x = map2multiclass(vol_label_x)
        vol_label_u = map2multiclass(vol_label_u)

        label_size = np.prod(list(vol_x_original_label_rfield.shape))
        vol_x_vendor_acc.append(
            (torch.sum(torch.tensor(vol_label_x == vol_x_original_label_rfield.squeeze())) / label_size).item()
        )
        vol_u_vendor_acc.append(
            (torch.sum(torch.tensor(vol_label_u == vol_x_original_label_rfield.squeeze())) / label_size).item()
        )

        if args.realfake_coef > 0:
            vol_x_realfake_acc.append(
                (torch.sum(
                    (torch.sigmoid(vol_real_label_x) > 0.5) == torch.ones_like(vol_real_label_x)
                ) / label_size).item()
            )
            vol_u_realfake_acc.append(
                (torch.sum(
                    (torch.sigmoid(vol_fake_label_u) > 0.5) == torch.zeros_like(vol_real_label_x)
                ) / label_size).item()
            )

        # --- Segmentator metrics ---
        set_grad([segmentator], False)
        segmentator.eval()
        pred_x = segmentator(vol_x)
        for mask_index, mask in enumerate(inestable_mask):
            if mask is not None:
                vol_x_iou.append(evaluate_segmentation(pred_x.detach()[mask_index], mask.squeeze()))
                vol_u_iou.append(evaluate_segmentation(pred_u.detach()[mask_index], mask.squeeze()))

    logging = {}

    # --- Plot examples ---
    if args.plot_examples:
        vendors_transformed_samples = []
        with torch.no_grad():
            for vendor_samples in vendors_samples:
                vendors_transformed_samples.append(
                    generator(vendor_samples).data.cpu().numpy()[:, 0, ...]
                )
        generated_samples = plot_save_generated_vendor_list(
            vendors_transformed_samples, os.path.join(args.output_dir, "generated_samples", f"epoch_{epoch}.jpg")
        )
        logging["Generated Examples"] = generated_samples

    vol_x_vendor_acc, vol_u_vendor_acc = np.mean(vol_x_vendor_acc), np.mean(vol_u_vendor_acc)
    dis_metrics = f"X Vendor Acc: {vol_x_vendor_acc:.4f} | U Vendor Acc: {vol_u_vendor_acc:.4f}"
    logging["Vendor X Acc"] = vol_x_vendor_acc
    logging["Vendor U Acc"] = vol_u_vendor_acc

    if args.realfake_coef > 0:
        vol_x_realfake_acc = np.mean(vol_x_realfake_acc)
        vol_u_realfake_acc = np.mean(vol_u_realfake_acc)
        dis_metrics += f" | X RealFake Acc: {vol_x_realfake_acc:.4f} | U RealFake Acc: {vol_u_realfake_acc:.4f}"
        logging["RealFake X Acc"] = vol_x_realfake_acc
        logging["RealFake U Acc"] = vol_u_realfake_acc

    seg_metrics = f" Vol X IOU: {np.mean(vol_x_iou):.4f} | Vol U IOU: {np.mean(vol_u_iou):.4f}"
    logging["Vol X IOU"] = np.mean(vol_x_iou)
    logging["Vol U IOU"] = np.mean(vol_u_iou)

    epoch_gen_loss, epoch_dis_loss = np.mean(epoch_gen_loss), np.mean(epoch_dis_loss)
    print(
        f"Epoch {epoch} | Gen Loss: {epoch_gen_loss:.4f} | Dis Loss: {epoch_dis_loss:.4f} "
        f"| {dis_metrics} | {seg_metrics}"
    )

    logging["Generator Loss"] = epoch_gen_loss
    logging["Discriminator Loss"] = epoch_dis_loss
    logging["Segmentator Learning Rate"] = g_optimizer.param_groups[0]['lr']
    logging["Generator Learning Rate"] = g_optimizer.param_groups[1]['lr']
    logging["Discriminator Learning Rate"] = d_optimizer.param_groups[0]['lr']

    # ---- Checkpoint ----
    torch.save(
        {
            'epoch': epoch + 1,
            'segmentator': segmentator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator': generator.state_dict(),
            'd_optimizer': d_optimizer.state_dict(),
            'g_optimizer': g_optimizer.state_dict()
        },
        f'{args.output_dir}/checkpoint.pt'
    )

    # Intermediate losses
    logging["Discriminator RealFake Loss"] = np.mean(epoch_dis_realfake_loss)
    logging["Discriminator X-VendorLabel Loss"] = np.mean(epoch_dis_xlabel_loss)
    logging["Discriminator U-VendorLabel Loss"] = np.mean(epoch_dis_ulabel_loss)
    logging["Segmentator X-Task Loss"] = np.mean(epoch_seg_xtask_loss)
    logging["Segmentator U-Task Loss"] = np.mean(epoch_seg_utask_loss)
    logging["Generator Cycle Loss"] = np.mean(epoch_gen_cycle_loss)
    logging["Generator U-VendorLabel Loss"] = np.mean(epoch_gen_ulabel_loss)
    logging["Generator U-Fake Loss"] = np.mean(epoch_gen_ufake_loss)

    # Logging

    wandb.log(logging)
    wandb.save(f'{args.output_dir}/checkpoint.pt')

    # -- Update learning rates --
    g_lr_scheduler.step()
    d_lr_scheduler.step()

wandb.finish()
