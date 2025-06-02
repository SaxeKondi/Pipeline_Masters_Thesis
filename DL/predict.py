import logging
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

# ------------------------
# Configuration Section
# ------------------------
config = {
    'model_path': './models/atomic-frost-9/checkpoint_epoch5.pth',
    'input_dir': './data/val/imgs/',  # Directory containing input images
    'output_dir': './results_temp/',   # Where to save output masks
    'visualize': False,
    'no_save': False,
    'mask_threshold': 0.5,
    'scale': 1.0,
    'bilinear': False,
    'classes': 2
}

def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    input_dir = Path(config['input_dir'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.jpeg'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {config["model_path"]}')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=config['classes'], bilinear=config['bilinear'])
    net.to(device=device)

    state_dict = torch.load(config['model_path'], map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for input_path in input_files:
        logging.info(f'Predicting image {input_path} ...')
        img = Image.open(input_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        mask = predict_img(
            net=net,
            full_img=img,
            scale_factor=config['scale'],
            out_threshold=config['mask_threshold'],
            device=device
        )

        if not config['no_save']:
            output_path = output_dir / f'{input_path.stem}_OUT.png'
            result = mask_to_image(mask, mask_values)
            result.save(output_path)
            logging.info(f'Mask saved to {output_path}')

        if config['visualize']:
            logging.info(f'Visualizing results for image {input_path}, close to continue...')
            plot_img_and_mask(img, mask)