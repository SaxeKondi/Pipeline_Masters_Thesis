import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

# ------------------------
# Configuration Section
# ------------------------
config = {
    'epochs': 20,
    'batch_size': 1,
    'learning_rate': 1e-6,
    'load': False,
    'scale': 1.0,
    'val': 10.2,
    'amp': True,
    'bilinear': False,
    'classes': 2
}

# Paths
dir_img = Path('./data/train/imgs/')
dir_mask = Path('./data/train/masks/')
dir_checkpoint = Path('./checkpoints/')

def train_model(
        model,
        device,
        epochs: int = 20,
        batch_size: int = 1,
        learning_rate: float = 1e-6,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # Prepare dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # Split data
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    # Compute class weights
    bg = weed = 0
    for idx in train_set.indices:
        mask = dataset[idx]['mask']
        bg += torch.sum(mask == 0)
        weed += torch.sum(mask == 1)
    total = bg + weed

    class_weights = torch.tensor([
        1,
        5
    ], dtype=torch.float, device=device)

    # class_weights = torch.tensor([
    #     total / (2 * bg),
    #     total / (2 * weed)
    # ], dtype=torch.float, device=device)

    print(class_weights)

    # Data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=False, **loader_args)

    # Initialize W&B
    experiment = wandb.init(project='U-Net-6-DiceLossOnly', resume='allow', anonymous='must')
    experiment.config.update(config)

    logging.info(
        f"Starting training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, "
        f"n_train={n_train}, n_val={n_val}, amp={amp}"
    )

    # Optimizer, scheduler, scaler, criterion
    optimizer = optim.RMSprop(
        model.parameters(), lr=learning_rate,
        weight_decay=weight_decay, momentum=momentum, foreach=True
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = (
        nn.CrossEntropyLoss(weight=class_weights)
        if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    )

    global_step = 0
    train_batch_count = 0
    for epoch in range(1, epochs + 1):
        model.train()
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = batch['mask'].to(device=device, dtype=torch.long)

                # Training step
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    preds = model(images)
                    if model.n_classes == 1:
                        train_loss = criterion(preds.squeeze(1), true_masks.float())
                        train_loss += dice_loss(
                            F.sigmoid(preds.squeeze(1)), true_masks.float(), multiclass=False
                        )
                    else:
                        train_loss = criterion(preds, true_masks)
                        train_loss += dice_loss(
                            F.softmax(preds, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0,3,1,2).float(),
                            multiclass=True
                        )
                scaler.scale(train_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                train_batch_count += 1
                # Log training loss every batch
                experiment.log({
                    'train_loss': train_loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })

                # Every 10 training batches, compute full validation metrics
                if train_batch_count % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_loss_total = 0.0
                        # Compute val loss + dice
                        for vbatch in val_loader:
                            vimg = vbatch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                            vtrue = vbatch['mask'].to(device=device, dtype=torch.long)
                            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                                vpred = model(vimg)
                                if model.n_classes == 1:
                                    vloss = criterion(vpred.squeeze(1), vtrue.float())
                                    vloss += dice_loss(
                                        F.sigmoid(vpred.squeeze(1)), vtrue.float(), multiclass=False
                                    )
                                else:
                                    # vloss = criterion(vpred, vtrue)
                                    vloss = dice_loss(
                                        F.softmax(vpred, dim=1).float(),
                                        F.one_hot(vtrue, model.n_classes).permute(0,3,1,2).float(),
                                        multiclass=True
                                    )
                            val_loss_total += vloss.item() * vimg.size(0)
                        avg_val_loss = val_loss_total / n_val
                        # Compute val dice
                        val_score = evaluate(model, val_loader, device, amp)
                        # Step scheduler
                        scheduler.step(val_score)
                        # Log validation metrics and images
                        logs = {
                            'val_loss': avg_val_loss,
                            'val_dice': val_score,
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'step': global_step,
                            'epoch': epoch
                        }
                        # Log example images and masks
                        logs['images'] = wandb.Image(vimg[0].cpu())
                        logs['masks_true'] = wandb.Image(vtrue[0].float().cpu())
                        if model.n_classes > 1:
                            pred_vis = vpred.argmax(dim=1)[0].float().cpu()
                        else:
                            pred_vis = (F.sigmoid(vpred.squeeze(1)) > 0.5)[0].float().cpu()
                        logs['masks_pred'] = wandb.Image(pred_vis)
                        experiment.log(logs)
                    model.train()

                pbar.set_postfix({
                    'train_loss': train_loss.item(),
                    **({'val_loss': avg_val_loss, 'val_dice': val_score} if train_batch_count % 10 == 0 else {})
                })
                pbar.update(images.size(0))


        # Save checkpoint
        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            state = model.state_dict()
            state['mask_values'] = dataset.mask_values
            torch.save(state, dir_checkpoint / f'checkpoint_epoch{epoch}.pth')
            logging.info(f'Checkpoint saved: epoch {epoch}')

    wandb.finish()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=3, n_classes=config['classes'], bilinear=config['bilinear'])
    model.to(memory_format=torch.channels_last).to(device=device)

    if config['load']:
        state = torch.load(config['load'], map_location=device)
        state.pop('mask_values', None)
        model.load_state_dict(state)
        logging.info(f"Loaded model from {config['load']}")

    train_model(
        model=model,
        device=device,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        val_percent=config['val'] / 100,
        img_scale=config['scale'],
        amp=config['amp']
    )
