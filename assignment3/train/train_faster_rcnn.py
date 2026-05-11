import torch
import argparse
import os
import numpy as np
import yaml
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.faster_rcnn import MaskRCNN
from tqdm import tqdm
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch):
    """
    Custom collate: handles variable-length mask lists per image.
    batch is a list of (im_tensor, target, fname) tuples.
    """
    images = torch.stack([b[0] for b in batch])
    fnames = [b[2] for b in batch]
    # Stack bboxes and labels normally (batch size is always 1 so this is fine)
    targets = {
        'bboxes': torch.stack([b[1]['bboxes'] for b in batch]),
        'labels': torch.stack([b[1]['labels'] for b in batch]),
        # masks stays as a plain list-of-lists — not stacked (variable size)
        'masks': [b[1]['masks'] for b in batch],
    }
    return images, targets, fnames


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc = VOCDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'],
                     seg_dir=dataset_config['seg_train_path'],
                     load_masks=True)
    train_dataset = DataLoader(voc,
                               batch_size=1,
                               shuffle=True,
                               num_workers=4,
                               collate_fn=collate_fn)

    faster_rcnn_model = MaskRCNN(model_config,
                                 num_classes=dataset_config['num_classes'])
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=filter(lambda p: p.requires_grad,
                                              faster_rcnn_model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)

    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        mask_losses = []
        optimizer.zero_grad()

        for im, target, fname in tqdm(train_dataset):
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            # Move masks to device if present (unpack from batch dimension)
            if target['masks'] is not None and len(target['masks']) > 0:
                target['masks'] = [m.to(device) for m in target['masks'][0]]
            else:
                target['masks'] = []

            rpn_output, frcnn_output = faster_rcnn_model(im, target)

            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
            mask_loss = frcnn_output.get('mask_loss', torch.tensor(0.0, device=device))
            loss = rpn_loss + frcnn_loss + mask_loss

            rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())
            mask_losses.append(mask_loss.item())

            loss = loss / acc_steps
            loss.backward()
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1

        print('Finished epoch {}'.format(i))
        optimizer.step()
        optimizer.zero_grad()
        torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                train_config['ckpt_name']))
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        loss_output += ' | Mask Loss : {:.4f}'.format(np.mean(mask_losses))
        print(loss_output)
        scheduler.step()
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    train(args)