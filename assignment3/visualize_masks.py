"""
Quick script to generate detection + mask visualizations for the report.
Runs on 10 random test images, saves to samples_mask/
Usage: PYTHONPATH=. python visualize_masks.py
"""
import torch
import cv2
import numpy as np
import os
import random
import yaml
import sys
sys.path.insert(0, '.')
from src.faster_rcnn import MaskRCNN
from dataset.voc import VOCDataset

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)

with open('config/voc.yaml') as f:
    config = yaml.safe_load(f)

dataset_config = config['dataset_params']
model_config = config['model_params']
train_config = config['train_params']

# Load dataset (no masks needed for inference)
voc = VOCDataset('test',
                 im_dir=dataset_config['im_test_path'],
                 ann_dir=dataset_config['ann_test_path'])

# Load model
model = MaskRCNN(model_config, num_classes=dataset_config['num_classes'])
ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
model.to(device)
model.roi_head.low_score_threshold = 0.5

os.makedirs('samples_mask', exist_ok=True)

# Colour palette for masks — one per class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(21, 3), dtype=np.uint8)

for sample_count in range(10):
    idx = random.randint(0, len(voc) - 1)
    im_tensor, target, fname = voc[idx]
    im_input = im_tensor.unsqueeze(0).float().to(device)

    with torch.no_grad():
        rpn_output, frcnn_output = model(im_input, None)

    boxes  = frcnn_output['boxes']
    labels = frcnn_output['labels']
    scores = frcnn_output['scores']
    masks  = frcnn_output.get('masks', [])

    # Read original image for drawing
    im = cv2.imread(fname)
    if im is None:
        print(f'Could not read {fname}, skipping')
        continue
    overlay = im.copy()

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = [int(v) for v in box.cpu().numpy()]
        cls = label.item()
        color = [int(c) for c in COLORS[cls]]
        label_name = voc.idx2label[cls]

        # Draw bounding box
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

        # Draw label + score
        text = f'{label_name}: {score:.2f}'
        cv2.putText(im, text, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw mask if available
        if i < len(masks):
            mask = masks[i]  # (H, W) bool tensor at feature map size
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy().astype(np.uint8)
            else:
                mask_np = np.array(mask, dtype=np.uint8)

            # Resize mask to original image size
            mask_resized = cv2.resize(mask_np, (im.shape[1], im.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

            # Apply coloured overlay where mask is active
            colored_mask = np.zeros_like(im)
            colored_mask[mask_resized == 1] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

    # Blend mask overlay with image
    result = cv2.addWeighted(im, 0.7, overlay, 0.3, 0)
    out_path = f'samples_mask/output_{sample_count}.jpg'
    cv2.imwrite(out_path, result)
    print(f'Saved {out_path} — {len(boxes)} detections')

print('Done. Check samples_mask/')