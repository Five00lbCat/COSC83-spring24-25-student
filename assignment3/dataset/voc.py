import glob
import os
import random

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
import numpy as np


def load_images_and_anns(im_dir, ann_dir, label2idx):
    im_infos = []
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, '*.xml'))):
        im_info = {}
        im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
        im_info['filename'] = os.path.join(im_dir, '{}.jpg'.format(im_info['img_id']))
        ann_info = ET.parse(ann_file)
        root = ann_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        im_info['width'] = width
        im_info['height'] = height
        detections = []
        for obj in ann_info.findall('object'):
            det = {}
            label = label2idx[obj.find('name').text]
            bbox_info = obj.find('bndbox')
            bbox = [
                int(float(bbox_info.find('xmin').text)) - 1,
                int(float(bbox_info.find('ymin').text)) - 1,
                int(float(bbox_info.find('xmax').text)) - 1,
                int(float(bbox_info.find('ymax').text)) - 1
            ]
            det['label'] = label
            det['bbox'] = bbox
            detections.append(det)
        im_info['detections'] = detections
        im_infos.append(im_info)
    print('Total {} images found'.format(len(im_infos)))
    return im_infos


# VOC segmentation colour palette: index -> (R, G, B)
# Instance segmentation PNGs use a fixed palette; each distinct colour = one object instance.
# Index 0 = background, 255 = boundary/ignore.
def _voc_palette():
    """Return the 256-entry VOC colour palette as a (256, 3) array."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i] = [r, g, b]
    return palette

_PALETTE = _voc_palette()


def load_seg_masks(seg_dir, img_id, num_objects):
    """
    Load the instance segmentation PNG for img_id and return a list of
    binary masks (one per object), in the order objects appear by colour index.

    VOC SegmentationObject PNGs are palette-indexed images where each
    unique colour index (1, 2, 3, …) corresponds to one object instance.
    Index 0 = background, 255 = boundary (ignore).

    Returns a list of (H, W) bool tensors, length == num_objects.
    If the seg file doesn't exist, returns None.
    """
    seg_path = os.path.join(seg_dir, '{}.png'.format(img_id))
    if not os.path.exists(seg_path):
        return None

    seg_img = Image.open(seg_path)
    seg_arr = np.array(seg_img, dtype=np.uint8)  # palette-indexed (H, W)

    # Collect unique instance indices (ignore 0=bg, 255=boundary)
    instance_indices = sorted([v for v in np.unique(seg_arr) if 0 < v < 255])

    masks = []
    for idx in instance_indices:
        mask = (seg_arr == idx)
        masks.append(torch.as_tensor(mask, dtype=torch.bool))

    # Pad or trim to match num_objects (annotation order may differ slightly)
    while len(masks) < num_objects:
        # If fewer seg instances than bbox annotations, pad with empty masks
        h, w = seg_arr.shape
        masks.append(torch.zeros(h, w, dtype=torch.bool))
    masks = masks[:num_objects]

    return masks


class VOCDataset(Dataset):
    def __init__(self, split, im_dir, ann_dir, seg_dir=None, load_masks=False):
        """
        Args:
            split:      'train' or 'test'
            im_dir:     path to JPEGImages
            ann_dir:    path to Annotations
            seg_dir:    path to SegmentationObject (required when load_masks=True)
            load_masks: if True, load instance segmentation masks for Mask R-CNN
        """
        self.split = split
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        self.seg_dir = seg_dir
        self.load_masks = load_masks and (seg_dir is not None)

        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(im_dir, ann_dir, self.label2idx)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info['filename'])
        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = {}
        targets['bboxes'] = torch.as_tensor(
            [detection['bbox'] for detection in im_info['detections']]
        )
        targets['labels'] = torch.as_tensor(
            [detection['label'] for detection in im_info['detections']]
        )

        # --- Segmentation masks (Mask R-CNN) ---
        if self.load_masks:
            masks = load_seg_masks(
                self.seg_dir,
                im_info['img_id'],
                num_objects=len(im_info['detections'])
            )
            targets['masks'] = masks if masks is not None else []
        else:
            targets['masks'] = []

        # --- Horizontal flip augmentation ---
        if to_flip:
            im_w = im_tensor.shape[-1]
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2 - x1
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])
            # Also flip the masks
            if targets['masks'] is not None:
                targets['masks'] = [m.flip(-1) for m in targets['masks']]

        return im_tensor, targets, im_info['filename']