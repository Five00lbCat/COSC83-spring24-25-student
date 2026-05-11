import torch
import torch.nn as nn
import torchvision
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ======================================================================
# Utility Functions (Already Implemented)
# ======================================================================

def get_iou(boxes1, boxes2):
    """Compute IoU between box sets (N x 4) and (M x 4)"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # (N, M)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])   # (N, M)
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2]) # (N, M)
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])# (N, M)
    
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1[:, None] + area2 - intersection_area
    iou = intersection_area / union
    return iou


def boxes_to_transformation_targets(ground_truth_boxes, anchors_or_proposals):
    """Convert bbox coordinates to regression targets (tx, ty, tw, th)"""
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * widths
    center_y = anchors_or_proposals[:, 1] + 0.5 * heights
    
    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights
    
    targets_dx = (gt_center_x - center_x) / widths
    targets_dy = (gt_center_y - center_y) / heights
    targets_dw = torch.log(gt_widths / widths)
    targets_dh = torch.log(gt_heights / heights)
    
    regression_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return regression_targets


def apply_regression_pred_to_anchors_or_proposals(box_transform_pred, anchors_or_proposals):
    """Apply predicted transformations to anchors/proposals to get predicted boxes"""
    box_transform_pred = box_transform_pred.reshape(box_transform_pred.size(0), -1, 4)
    
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * w
    center_y = anchors_or_proposals[:, 1] + 0.5 * h
    
    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]
    
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))
    
    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]
    
    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h
    
    pred_boxes = torch.stack((pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2), dim=2)
    return pred_boxes


def sample_positive_negative(labels, positive_count, total_count):
    """Sample positive and negative examples for training"""
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]
    
    num_pos = min(positive.numel(), positive_count)
    num_neg = min(negative.numel(), total_count - num_pos)
    
    perm_positive_idxs = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm_negative_idxs = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
    
    pos_idxs = positive[perm_positive_idxs]
    neg_idxs = negative[perm_negative_idxs]
    
    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_pos_idx_mask[pos_idxs] = True
    sampled_neg_idx_mask[neg_idxs] = True
    
    return sampled_neg_idx_mask, sampled_pos_idx_mask


def clamp_boxes_to_image_boundary(boxes, image_shape):
    """Clip boxes to stay within image boundaries"""
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]
    
    height, width = image_shape[-2:]
    boxes_x1 = boxes_x1.clamp(min=0, max=width)
    boxes_x2 = boxes_x2.clamp(min=0, max=width)
    boxes_y1 = boxes_y1.clamp(min=0, max=height)
    boxes_y2 = boxes_y2.clamp(min=0, max=height)
    
    boxes = torch.cat((
        boxes_x1[..., None],
        boxes_y1[..., None],
        boxes_x2[..., None],
        boxes_y2[..., None]),
        dim=-1)
    return boxes


def transform_boxes_to_original_size(boxes, new_size, original_size):
    """Scale bounding boxes back to original image dimensions"""
    ratios = [
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


# ======================================================================
# Part 2: Region Proposal Network (20%)
# ======================================================================

class RegionProposalNetwork(nn.Module):
    """Region Proposal Network for Faster R-CNN"""
    
    def __init__(self, in_channels, scales, aspect_ratios, model_config):
        super(RegionProposalNetwork, self).__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.low_iou_threshold = model_config['rpn_bg_threshold']
        self.high_iou_threshold = model_config['rpn_fg_threshold']
        self.rpn_nms_threshold = model_config['rpn_nms_threshold']
        self.rpn_batch_size = model_config['rpn_batch_size']
        self.rpn_pos_count = int(model_config['rpn_pos_fraction'] * self.rpn_batch_size)
        self.rpn_topk = model_config['rpn_train_topk'] if self.training else model_config['rpn_test_topk']
        self.rpn_prenms_topk = model_config['rpn_train_prenms_topk'] if self.training else model_config['rpn_test_prenms_topk']
        
        # Number of anchors per spatial location = num_scales * num_aspect_ratios
        self.num_anchors = len(scales) * len(aspect_ratios)
        
        # 3x3 conv shared layer
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # 1x1 classification conv: outputs objectness score per anchor
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1)
        # 1x1 regression conv: outputs 4 deltas per anchor
        self.reg_layer = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=1)
        
        # Weight initialization
        for layer in [self.rpn_conv, self.cls_layer, self.reg_layer]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def generate_anchors(self, image, feat):
        """Generate anchors for all feature map locations with all scales and aspect ratios"""
        # Stride = ratio of input image size to feature map size
        feat_h, feat_w = feat.shape[-2:]
        img_h, img_w = image.shape[-2:]
        stride_h = img_h // feat_h
        stride_w = img_w // feat_w
        
        # Build base anchors at (0,0) for all scale/aspect_ratio combos
        base_anchors = []
        for scale in self.scales:
            for ar in self.aspect_ratios:
                # ar = h/w  =>  w = sqrt(area/ar), h = w*ar
                w = scale / math.sqrt(ar)
                h = scale * math.sqrt(ar)
                # Center at 0
                base_anchors.append([
                    -w / 2, -h / 2, w / 2, h / 2
                ])
        base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=feat.device)
        # shape: (num_anchors, 4)
        
        # Create grid of shift values
        shifts_x = torch.arange(0, feat_w, device=feat.device, dtype=torch.float32) * stride_w
        shifts_y = torch.arange(0, feat_h, device=feat.device, dtype=torch.float32) * stride_h
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        
        # shifts shape: (num_locations, 4) — same delta for x1,x2 and y1,y2
        shifts = torch.stack([shifts_x, shifts_y, shifts_x, shifts_y], dim=1)
        # shape: (num_locations, 4)
        
        # Combine: (num_locations, 1, 4) + (1, num_anchors, 4) => (num_locations, num_anchors, 4)
        all_anchors = shifts[:, None, :] + base_anchors[None, :, :]
        all_anchors = all_anchors.reshape(-1, 4)
        # shape: (num_locations * num_anchors, 4)
        return all_anchors
    
    def assign_targets_to_anchors(self, anchors, gt_boxes):
        """Assign ground truth boxes and labels to anchors based on IoU"""
        # gt_boxes shape: (num_gt, 4)
        iou_matrix = get_iou(anchors, gt_boxes)  # (N_anchors, N_gt)
        
        # For each anchor, best matching GT index and IoU
        best_gt_iou_per_anchor, best_gt_idx_per_anchor = iou_matrix.max(dim=1)
        
        # Initialize all labels as -1 (ignore)
        labels = torch.full((anchors.shape[0],), -1, dtype=torch.float32, device=anchors.device)
        
        # Negative: below low threshold
        labels[best_gt_iou_per_anchor < self.low_iou_threshold] = 0
        
        # Positive: above high threshold
        labels[best_gt_iou_per_anchor >= self.high_iou_threshold] = 1
        
        # Also mark the best anchor for each GT box as positive (ensures every GT has a match)
        best_anchor_iou_per_gt, best_anchor_idx_per_gt = iou_matrix.max(dim=0)
        labels[best_anchor_idx_per_gt] = 1
        
        # Matched GT boxes for all anchors
        matched_gt_boxes = gt_boxes[best_gt_idx_per_anchor]
        
        return labels, matched_gt_boxes

    def filter_proposals(self, proposals, cls_scores, image_shape):
        """Filter proposals using NMS and score thresholds"""
        # proposals: (N, num_anchors, 4) — squeeze to (N_proposals, 4)
        proposals = proposals.reshape(-1, 4)
        cls_scores = cls_scores.reshape(-1)
        
        # Convert logits to probabilities for sorting
        objectness_scores = cls_scores.detach()
        
        # Pre-NMS top-k selection
        prenms_topk = min(self.rpn_prenms_topk if self.training else
                          self.rpn_prenms_topk, objectness_scores.shape[0])
        top_k_idx = objectness_scores.topk(prenms_topk).indices
        proposals = proposals[top_k_idx]
        cls_scores = cls_scores[top_k_idx]
        objectness_scores = objectness_scores[top_k_idx]
        
        # Clamp proposals to image boundary
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)
        
        # Remove small boxes (width or height < 1 pixel)
        widths = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        keep = (widths >= 1) & (heights >= 1)
        proposals = proposals[keep]
        cls_scores = cls_scores[keep]
        objectness_scores = objectness_scores[keep]
        
        # NMS
        keep_nms = torchvision.ops.nms(proposals, objectness_scores, self.rpn_nms_threshold)
        
        # Post-NMS top-k
        post_nms_topk = self.rpn_topk if self.training else self.rpn_topk
        keep_nms = keep_nms[:post_nms_topk]
        
        proposals = proposals[keep_nms]
        cls_scores = cls_scores[keep_nms]
        
        return proposals, cls_scores
    
    def forward(self, image, feat, target=None):
        """Forward pass for RPN"""
        # 1. Shared conv + ReLU
        t = torch.relu(self.rpn_conv(feat))
        
        # 2. Classification and regression predictions
        cls_scores = self.cls_layer(t)    # (B, num_anchors, H, W)
        box_transform_pred = self.reg_layer(t)  # (B, num_anchors*4, H, W)
        
        # 3. Generate anchors
        anchors = self.generate_anchors(image, feat)  # (N_anchors_total, 4)
        
        # 4. Reshape predictions: (B, num_anchors, H, W) => (B, H*W*num_anchors)
        batch_size = feat.shape[0]
        # cls: (B, num_anchors, H, W) => (B, H, W, num_anchors) => (B, -1)
        cls_scores_flat = cls_scores.permute(0, 2, 3, 1).reshape(batch_size, -1)
        # reg: (B, num_anchors*4, H, W) => (B, H, W, num_anchors*4) => (B, -1, 4)
        box_transform_flat = box_transform_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        # 5. Apply regression to anchors to get proposals
        proposals = apply_regression_pred_to_anchors_or_proposals(
            box_transform_flat[0], anchors
        )  # (N_anchors, num_classes, 4) but we have 1 class => squeeze
        proposals = proposals[:, 0, :]  # (N_anchors, 4)
        
        # 6. Filter proposals
        proposals, scores = self.filter_proposals(proposals, cls_scores_flat[0], image.shape)
        
        output = {
            'proposals': proposals,
            'scores': scores,
        }
        
        if target is not None and self.training:
            # gt_boxes: (1, N_gt, 4) => (N_gt, 4)
            gt_boxes = target['bboxes'][0]  # (N_gt, 4)
            
            # 7. Assign targets to anchors
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, gt_boxes)
            
            # 8. Sample positive and negative anchors
            sampled_neg_mask, sampled_pos_mask = sample_positive_negative(
                labels, self.rpn_pos_count, self.rpn_batch_size
            )
            sampled_mask = sampled_pos_mask | sampled_neg_mask
            
            # 9. Classification loss (binary cross entropy with logits)
            # Only on sampled anchors
            sampled_cls_scores = cls_scores_flat[0][sampled_mask]
            sampled_labels = labels[sampled_mask]
            rpn_cls_loss = nn.functional.binary_cross_entropy_with_logits(
                sampled_cls_scores,
                sampled_labels.float()
            )
            
            # Regression loss (smooth L1) only on positive anchors
            pos_anchors = anchors[sampled_pos_mask]
            pos_matched_gt = matched_gt_boxes[sampled_pos_mask]
            
            if pos_anchors.shape[0] > 0:
                reg_targets = boxes_to_transformation_targets(pos_matched_gt, pos_anchors)
                # box_transform_flat: (B, N_anchors, 4)
                pos_box_preds = box_transform_flat[0][sampled_pos_mask]
                rpn_loc_loss = nn.functional.smooth_l1_loss(pos_box_preds, reg_targets)
            else:
                rpn_loc_loss = torch.tensor(0.0, device=feat.device)
            
            output['rpn_classification_loss'] = rpn_cls_loss
            output['rpn_localization_loss'] = rpn_loc_loss
        
        return output


# ======================================================================
# Part 3: RoI Feature Extraction and Part 4: Detection Head (40%)
# ======================================================================

class ROIHead(nn.Module):
    """ROI head for final classification and box refinement"""
    
    def __init__(self, model_config, num_classes, in_channels):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.roi_batch_size = model_config['roi_batch_size']
        self.roi_pos_count = int(model_config['roi_pos_fraction'] * self.roi_batch_size)
        self.iou_threshold = model_config['roi_iou_threshold']
        self.low_bg_iou = model_config['roi_low_bg_iou']
        self.nms_threshold = model_config['roi_nms_threshold']
        self.topK_detections = model_config['roi_topk_detections']
        self.low_score_threshold = model_config['roi_score_threshold']
        self.pool_size = model_config['roi_pool_size']
        self.fc_inner_dim = model_config['fc_inner_dim']
        
        # Two FC layers for feature transformation
        # After RoI pooling: in_channels * pool_size * pool_size
        roi_feat_dim = in_channels * self.pool_size * self.pool_size
        self.fc1 = nn.Linear(roi_feat_dim, self.fc_inner_dim)
        self.fc2 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        
        # Classification: num_classes scores (background + object classes)
        self.cls_layer = nn.Linear(self.fc_inner_dim, num_classes)
        # Box regression: 4 deltas per class
        self.reg_layer = nn.Linear(self.fc_inner_dim, num_classes * 4)
        
        # Weight initialization
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.cls_layer.weight, std=0.01)
        nn.init.constant_(self.cls_layer.bias, 0)
        nn.init.normal_(self.reg_layer.weight, std=0.001)
        nn.init.constant_(self.reg_layer.bias, 0)
    
    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        """Assign ground truth boxes and labels to proposals based on IoU"""
        iou_matrix = get_iou(proposals, gt_boxes)  # (N_proposals, N_gt)
        
        # For each proposal, best matching GT
        best_gt_iou, best_gt_idx = iou_matrix.max(dim=1)
        
        # Initialize labels to -1 (ignore)
        labels = torch.full((proposals.shape[0],), -1, dtype=torch.long, device=proposals.device)
        
        # Background: IoU < low_bg_iou threshold — actually between low and high thresholds gets -1
        # Positive: IoU >= iou_threshold
        # Background (0): low_bg_iou <= IoU < iou_threshold
        # Ignore (-1): IoU < low_bg_iou (very low overlap, don't use)
        labels[best_gt_iou < self.low_bg_iou] = -1
        labels[(best_gt_iou >= self.low_bg_iou) & (best_gt_iou < self.iou_threshold)] = 0
        labels[best_gt_iou >= self.iou_threshold] = gt_labels[best_gt_idx[best_gt_iou >= self.iou_threshold]]
        
        # Get matched gt boxes for all proposals
        matched_gt_boxes = gt_boxes[best_gt_idx]
        
        return labels, matched_gt_boxes
    
    def forward(self, feat, proposals, image_shape, target):
        """Forward pass for ROI head"""
        if target is not None and self.training:
            # 1. Add GT boxes to proposals for better training coverage
            gt_boxes = target['bboxes'][0]   # (N_gt, 4)
            gt_labels = target['labels'][0]  # (N_gt,)
            proposals = torch.cat([proposals, gt_boxes], dim=0)
            
            # 2. Assign targets to proposals
            labels, matched_gt_boxes = self.assign_target_to_proposals(
                proposals, gt_boxes, gt_labels
            )
            
            # 3. Sample positive and negative proposals
            sampled_neg_mask, sampled_pos_mask = sample_positive_negative(
                labels, self.roi_pos_count, self.roi_batch_size
            )
            sampled_mask = sampled_pos_mask | sampled_neg_mask
            proposals = proposals[sampled_mask]
            labels = labels[sampled_mask]
            matched_gt_boxes = matched_gt_boxes[sampled_mask]
        
        # 4. Scale for RoI pooling: feat_size / image_size
        feat_h, feat_w = feat.shape[-2:]
        img_h, img_w = image_shape
        scale = min(feat_h / img_h, feat_w / img_w)
        
        # 5. RoI Pooling: extract fixed-size features per proposal
        # torchvision.ops.roi_pool expects boxes as (batch_idx, x1, y1, x2, y2)
        # Since batch_size=1, prepend zeros
        roi_indices = torch.zeros(proposals.shape[0], 1, device=feat.device)
        rois = torch.cat([roi_indices, proposals], dim=1)
        
        roi_feats = torchvision.ops.roi_pool(feat, rois, output_size=self.pool_size, spatial_scale=scale)
        # shape: (N_proposals, in_channels, pool_size, pool_size)
        
        # 6. Flatten and apply FC layers
        roi_feats = roi_feats.flatten(start_dim=1)
        roi_feats = torch.relu(self.fc1(roi_feats))
        roi_feats = torch.relu(self.fc2(roi_feats))
        
        # 7. Generate classification and regression predictions
        cls_scores = self.cls_layer(roi_feats)       # (N, num_classes)
        box_transform_pred = self.reg_layer(roi_feats)  # (N, num_classes * 4)
        
        if target is not None and self.training:
            # 8. Classification loss
            frcnn_cls_loss = nn.functional.cross_entropy(cls_scores, labels)
            
            # 9. Regression loss — only on positive proposals, using the correct class deltas
            pos_mask = sampled_pos_mask[sampled_mask] if sampled_mask is not None else (labels > 0)
            # Recompute using direct label comparison since we've already subsetted
            pos_mask = labels > 0
            
            if pos_mask.sum() > 0:
                pos_proposals = proposals[pos_mask]
                pos_gt_boxes = matched_gt_boxes[pos_mask]
                pos_labels = labels[pos_mask]
                
                reg_targets = boxes_to_transformation_targets(pos_gt_boxes, pos_proposals)
                
                # Select the regression predictions for the correct class
                box_transform_pos = box_transform_pred[pos_mask]
                # box_transform_pos: (N_pos, num_classes * 4)
                # Reshape to (N_pos, num_classes, 4) and select per-class predictions
                box_transform_pos = box_transform_pos.reshape(-1, self.num_classes, 4)
                box_transform_pos = box_transform_pos[torch.arange(pos_labels.shape[0]), pos_labels]
                # shape: (N_pos, 4)
                
                frcnn_loc_loss = nn.functional.smooth_l1_loss(box_transform_pos, reg_targets)
            else:
                frcnn_loc_loss = torch.tensor(0.0, device=feat.device)
            
            return {
                'frcnn_classification_loss': frcnn_cls_loss,
                'frcnn_localization_loss': frcnn_loc_loss,
            }
        
        else:
            # 9. Inference: apply regression to proposals
            # box_transform_pred: (N, num_classes * 4) => (N, num_classes, 4)
            pred_boxes = apply_regression_pred_to_anchors_or_proposals(
                box_transform_pred, proposals
            )  # (N, num_classes, 4)
            
            # Get class probabilities
            pred_scores = torch.softmax(cls_scores, dim=1)  # (N, num_classes)
            
            # Expand proposals and scores for all classes
            # For each proposal, consider all non-background classes
            # pred_boxes: (N, num_classes, 4)
            # pred_scores: (N, num_classes)
            
            all_boxes = []
            all_labels = []
            all_scores = []
            
            for class_idx in range(1, self.num_classes):  # skip background (0)
                class_scores = pred_scores[:, class_idx]  # (N,)
                class_boxes = pred_boxes[:, class_idx, :]  # (N, 4)
                
                # Clamp to image
                class_boxes = clamp_boxes_to_image_boundary(class_boxes, image_shape)
                
                # Keep only above score threshold
                keep = class_scores > self.low_score_threshold
                class_boxes = class_boxes[keep]
                class_scores = class_scores[keep]
                
                if class_scores.numel() == 0:
                    continue
                
                all_boxes.append(class_boxes)
                all_labels.append(torch.full((class_boxes.shape[0],), class_idx,
                                             dtype=torch.long, device=feat.device))
                all_scores.append(class_scores)
            
            if len(all_boxes) == 0:
                return {
                    'boxes': torch.zeros((0, 4), device=feat.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=feat.device),
                    'scores': torch.zeros(0, device=feat.device),
                }
            
            all_boxes = torch.cat(all_boxes, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            
            # 10. Filter predictions
            boxes, labels, scores = self.filter_predictions(all_boxes, all_labels, all_scores)
            
            return {
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
            }
    
    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        """Filter predictions by score, size, and NMS"""
        # 1. Remove small boxes
        widths = pred_boxes[:, 2] - pred_boxes[:, 0]
        heights = pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (widths >= 1) & (heights >= 1)
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]
        
        if pred_scores.numel() == 0:
            return pred_boxes, pred_labels, pred_scores
        
        # 2. Per-class NMS
        keep_nms = torchvision.ops.batched_nms(pred_boxes, pred_scores, pred_labels, self.nms_threshold)
        pred_boxes = pred_boxes[keep_nms]
        pred_labels = pred_labels[keep_nms]
        pred_scores = pred_scores[keep_nms]
        
        # 3. Sort by score and keep top-k
        sorted_idx = pred_scores.argsort(descending=True)
        sorted_idx = sorted_idx[:self.topK_detections]
        pred_boxes = pred_boxes[sorted_idx]
        pred_labels = pred_labels[sorted_idx]
        pred_scores = pred_scores[sorted_idx]
        
        return pred_boxes, pred_labels, pred_scores


# ======================================================================
# Part 5: Faster R-CNN Model (20%)
# ======================================================================

class FasterRCNN(nn.Module):
    """Faster R-CNN object detection model"""
    
    def __init__(self, model_config, num_classes):
        super(FasterRCNN, self).__init__()
        self.model_config = model_config
        
        # VGG16 backbone (provided)
        vgg16 = torchvision.models.vgg16(weights="DEFAULT")
        self.backbone = vgg16.features[:-1]
        
        # RPN
        backbone_out_channels = model_config['backbone_out_channels']
        self.rpn = RegionProposalNetwork(
            in_channels=backbone_out_channels,
            scales=model_config['scales'],
            aspect_ratios=model_config['aspect_ratios'],
            model_config=model_config,
        )
        
        # ROI Head
        self.roi_head = ROIHead(
            model_config=model_config,
            num_classes=num_classes,
            in_channels=backbone_out_channels,
        )
        
        # Freeze early backbone layers
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']
    
    def normalize_resize_image_and_boxes(self, image, bboxes=None):
        """Normalize and resize image, adjusting bboxes accordingly"""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        c, h, w = image.shape[-3:]
        
        image = image.float()
        mean = torch.as_tensor(self.image_mean, dtype=image.dtype, device=image.device)
        std = torch.as_tensor(self.image_std, dtype=image.dtype, device=image.device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        
        min_original_size = float(min((h, w)))
        max_original_size = float(max((h, w)))
        scale_factor_min = self.min_size / min_original_size
        
        if max_original_size * scale_factor_min > self.max_size:
            scale_factor = self.max_size / max_original_size
        else:
            scale_factor = scale_factor_min
        
        image = torch.nn.functional.interpolate(
            image, scale_factor=scale_factor, mode='bilinear',
            recompute_scale_factor=True, align_corners=False
        )
        
        if bboxes is not None:
            if bboxes.dim() == 2:
                ratios = [
                    torch.tensor(s, dtype=torch.float32, device=bboxes.device) /
                    torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                    for s, s_orig in zip(image.shape[-2:], (h, w))
                ]
                ratio_height, ratio_width = ratios
                xmin = bboxes[:, 0] * ratio_width
                ymin = bboxes[:, 1] * ratio_height
                xmax = bboxes[:, 2] * ratio_width
                ymax = bboxes[:, 3] * ratio_height
                bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
            
            elif bboxes.dim() == 3:
                ratios = [
                    torch.tensor(s, dtype=torch.float32, device=bboxes.device) /
                    torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                    for s, s_orig in zip(image.shape[-2:], (h, w))
                ]
                ratio_height, ratio_width = ratios
                xmin, ymin, xmax, ymax = bboxes.unbind(2)
                xmin = xmin * ratio_width
                xmax = xmax * ratio_width
                ymin = ymin * ratio_height
                ymax = ymax * ratio_height
                bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
        
        return image, bboxes
    
    def forward(self, image, target=None):
        """Forward pass for Faster R-CNN"""
        # 1. Save original image shape (H, W)
        original_image_size = (image.shape[-2], image.shape[-1])
        
        # 2. Normalize and resize image (and boxes during training)
        if target is not None:
            image, target['bboxes'] = self.normalize_resize_image_and_boxes(
                image, target['bboxes']
            )
        else:
            image, _ = self.normalize_resize_image_and_boxes(image, None)
        
        resized_image_size = (image.shape[-2], image.shape[-1])
        
        # 3. Extract features with backbone
        feat = self.backbone(image)
        
        # 4. Get region proposals from RPN
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']
        
        # 5. Process proposals with ROI head
        frcnn_output = self.roi_head(feat, proposals, resized_image_size, target)
        
        # 6. During inference, scale boxes back to original image size
        if target is None and 'boxes' in frcnn_output and frcnn_output['boxes'].shape[0] > 0:
            frcnn_output['boxes'] = transform_boxes_to_original_size(
                frcnn_output['boxes'],
                new_size=resized_image_size,
                original_size=original_image_size,
            )
        
        return rpn_output, frcnn_output


# ======================================================================
# BONUS: Mask R-CNN Extension
# ======================================================================

class MaskHead(nn.Module):
    """
    Mask branch: small FCN that predicts a binary mask per class.
    Input: RoI-Aligned features (N, C, mask_pool_size, mask_pool_size)
    Output: (N, num_classes, mask_pool_size*2, mask_pool_size*2)
    We upsample 2x so the mask has reasonable resolution.
    """
    def __init__(self, in_channels, num_classes, mask_pool_size=14):
        super(MaskHead, self).__init__()
        self.mask_pool_size = mask_pool_size
        layers = []
        # 4 conv layers with ReLU, then a deconv to upsample 2x
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        # Upsample 2x via transposed conv
        layers.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2))
        layers.append(nn.ReLU(inplace=True))
        # Final 1x1 conv: one mask per class
        layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))
        self.mask_fcn = nn.Sequential(*layers)

        # Weight init
        for m in self.mask_fcn:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mask_fcn(x)


class MaskRCNN(nn.Module):
    """
    Mask R-CNN: extends FasterRCNN by adding a mask branch.
    Uses RoI Align (torchvision.ops.roi_align) instead of RoI Pool
    for better spatial alignment, which matters for pixel-level masks.
    """
    def __init__(self, model_config, num_classes):
        super(MaskRCNN, self).__init__()
        self.model_config = model_config
        self.num_classes = num_classes

        # --- Backbone (VGG16, same as Faster R-CNN) ---
        vgg16 = torchvision.models.vgg16(weights="DEFAULT")
        self.backbone = vgg16.features[:-1]
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False

        # --- RPN (identical to Faster R-CNN) ---
        backbone_out_channels = model_config['backbone_out_channels']
        self.rpn = RegionProposalNetwork(
            in_channels=backbone_out_channels,
            scales=model_config['scales'],
            aspect_ratios=model_config['aspect_ratios'],
            model_config=model_config,
        )

        # --- Detection ROI Head (identical to Faster R-CNN) ---
        self.roi_head = ROIHead(
            model_config=model_config,
            num_classes=num_classes,
            in_channels=backbone_out_channels,
        )

        # --- Mask Head ---
        self.mask_pool_size = model_config.get('mask_pool_size', 14)
        self.mask_head = MaskHead(
            in_channels=backbone_out_channels,
            num_classes=num_classes,
            mask_pool_size=self.mask_pool_size,
        )

        # Image normalization (same as Faster R-CNN)
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']

    def normalize_resize_image_and_boxes(self, image, bboxes=None):
        """Identical to FasterRCNN — normalize + resize image and boxes."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        c, h, w = image.shape[-3:]
        image = image.float()
        mean = torch.as_tensor(self.image_mean, dtype=image.dtype, device=image.device)
        std = torch.as_tensor(self.image_std, dtype=image.dtype, device=image.device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        min_original_size = float(min(h, w))
        max_original_size = float(max(h, w))
        scale_factor = self.min_size / min_original_size
        if max_original_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_original_size
        image = torch.nn.functional.interpolate(
            image, scale_factor=scale_factor, mode='bilinear',
            recompute_scale_factor=True, align_corners=False
        )
        if bboxes is not None:
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=bboxes.device) /
                torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                for s, s_orig in zip(image.shape[-2:], (h, w))
            ]
            ratio_height, ratio_width = ratios
            if bboxes.dim() == 2:
                xmin, ymin, xmax, ymax = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            else:
                xmin, ymin, xmax, ymax = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
            xmin = xmin * ratio_width;  xmax = xmax * ratio_width
            ymin = ymin * ratio_height; ymax = ymax * ratio_height
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=-1)
        return image, bboxes

    def _resize_masks(self, masks, scale_factor):
        """Resize segmentation masks to match the resized image."""
        # masks: (N, H, W) binary tensors
        if masks is None:
            return None
        masks = masks.float().unsqueeze(1)  # (N, 1, H, W)
        masks = torch.nn.functional.interpolate(
            masks, scale_factor=scale_factor, mode='nearest'
        )
        return masks.squeeze(1)  # (N, H_new, W_new)

    def _extract_mask_targets(self, proposals, gt_boxes, gt_masks, labels):
        """
        For each positive proposal, crop the corresponding GT mask to the
        proposal region and resize to (mask_pool_size*2, mask_pool_size*2).
        Returns binary mask targets for the mask loss.
        """
        mask_size = self.mask_pool_size * 2
        mask_targets = []
        for i, (prop, label) in enumerate(zip(proposals, labels)):
            if label == 0:
                # Background — append dummy, won't be used in loss
                mask_targets.append(torch.zeros(mask_size, mask_size, device=proposals.device))
                continue
            # Find best matching GT box for this proposal
            iou = get_iou(prop.unsqueeze(0), gt_boxes)  # (1, N_gt)
            gt_idx = iou.argmax().item()
            gt_mask = gt_masks[gt_idx]  # (H_img, W_img) binary

            # Crop mask to the proposal bounding box
            x1, y1, x2, y2 = prop.long()
            x1 = x1.clamp(min=0); y1 = y1.clamp(min=0)
            x2 = x2.clamp(max=gt_mask.shape[1])
            y2 = y2.clamp(max=gt_mask.shape[0])
            cropped = gt_mask[y1:y2, x1:x2].float()

            if cropped.numel() == 0:
                mask_targets.append(torch.zeros(mask_size, mask_size, device=proposals.device))
                continue

            # Resize to mask output size
            cropped = cropped.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
            resized = torch.nn.functional.interpolate(
                cropped, size=(mask_size, mask_size), mode='bilinear', align_corners=False
            )
            mask_targets.append((resized.squeeze() > 0.5).float())

        return torch.stack(mask_targets, dim=0)  # (N, mask_size, mask_size)

    def forward(self, image, target=None):
        """
        Forward pass for Mask R-CNN.
        target (training only) should contain:
            'bboxes': (1, N_gt, 4)
            'labels': (1, N_gt)
            'masks':  list of (H, W) binary tensors, one per GT object
        """
        original_image_size = (image.shape[-2], image.shape[-1])

        # --- Preprocess ---
        if target is not None:
            image, target['bboxes'] = self.normalize_resize_image_and_boxes(
                image, target['bboxes']
            )
            # Resize masks to match resized image
            if 'masks' in target and target['masks'] is not None:
                orig_h, orig_w = original_image_size
                new_h, new_w = image.shape[-2:]
                scale_h = new_h / orig_h
                scale_w = new_w / orig_w
                # Resize each mask individually (they share the same scale)
                resized_masks = []
                for m in target['masks']:
                    m_f = m.float().unsqueeze(0).unsqueeze(0)
                    m_r = torch.nn.functional.interpolate(
                        m_f, size=(new_h, new_w), mode='nearest'
                    ).squeeze()
                    resized_masks.append(m_r)
                target['masks'] = resized_masks
        else:
            image, _ = self.normalize_resize_image_and_boxes(image, None)

        resized_image_size = (image.shape[-2], image.shape[-1])

        # --- Backbone ---
        feat = self.backbone(image)
        feat_h, feat_w = feat.shape[-2:]
        img_h, img_w = resized_image_size
        spatial_scale = min(feat_h / img_h, feat_w / img_w)

        # --- RPN ---
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']

        # --- Detection Head ---
        det_output = self.roi_head(feat, proposals, resized_image_size, target)

        # --- Mask Head ---
        if target is not None and self.training:
            # Use the sampled positive proposals from detection head for mask training.
            # Re-sample: get all proposals + GT, assign labels, sample positives
            gt_boxes  = target['bboxes'][0]
            gt_labels = target['labels'][0]
            gt_masks  = target.get('masks', None)

            all_proposals = torch.cat([proposals, gt_boxes], dim=0)
            labels, matched_gt_boxes = self.roi_head.assign_target_to_proposals(
                all_proposals, gt_boxes, gt_labels
            )
            _, pos_mask = sample_positive_negative(labels, self.roi_head.roi_pos_count, self.roi_head.roi_batch_size)

            pos_proposals = all_proposals[pos_mask]
            pos_labels    = labels[pos_mask]

            mask_loss = torch.tensor(0.0, device=feat.device)
            if pos_proposals.shape[0] > 0 and gt_masks is not None and len(gt_masks) > 0:
                # RoI Align on positive proposals (mask_pool_size for mask branch)
                roi_indices = torch.zeros(pos_proposals.shape[0], 1, device=feat.device)
                rois = torch.cat([roi_indices, pos_proposals], dim=1)
                mask_feats = torchvision.ops.roi_align(
                    feat, rois,
                    output_size=self.mask_pool_size,
                    spatial_scale=spatial_scale,
                    sampling_ratio=2,
                )
                # Forward through mask FCN
                mask_preds = self.mask_head(mask_feats)
                # mask_preds: (N_pos, num_classes, mask_size*2, mask_size*2)

                # Build mask targets
                mask_targets = self._extract_mask_targets(
                    pos_proposals, gt_boxes,
                    [m.to(feat.device) for m in gt_masks],
                    pos_labels
                )  # (N_pos, mask_size*2, mask_size*2)

                # Select predicted masks for the correct class
                pos_mask_preds = mask_preds[
                    torch.arange(pos_labels.shape[0], device=feat.device),
                    pos_labels
                ]  # (N_pos, mask_size*2, mask_size*2)

                # Binary cross-entropy mask loss (only on foreground proposals)
                fg = pos_labels > 0
                if fg.sum() > 0:
                    mask_loss = nn.functional.binary_cross_entropy_with_logits(
                        pos_mask_preds[fg], mask_targets[fg]
                    )

            det_output['mask_loss'] = mask_loss
            return rpn_output, det_output

        else:
            # --- Inference: generate masks for detected boxes ---
            boxes  = det_output.get('boxes', torch.zeros((0, 4), device=feat.device))
            labels = det_output.get('labels', torch.zeros(0, dtype=torch.long, device=feat.device))

            masks = []
            if boxes.shape[0] > 0:
                roi_indices = torch.zeros(boxes.shape[0], 1, device=feat.device)
                rois = torch.cat([roi_indices, boxes], dim=1)
                mask_feats = torchvision.ops.roi_align(
                    feat, rois,
                    output_size=self.mask_pool_size,
                    spatial_scale=spatial_scale,
                    sampling_ratio=2,
                )
                mask_preds = self.mask_head(mask_feats)  # (N, num_classes, H, W)
                # Select per-class masks and threshold
                for i, label in enumerate(labels):
                    m = torch.sigmoid(mask_preds[i, label])  # (H, W)
                    masks.append((m > 0.5).cpu())

            det_output['masks'] = masks

            # Scale boxes back to original image size
            if boxes.shape[0] > 0:
                det_output['boxes'] = transform_boxes_to_original_size(
                    boxes, new_size=resized_image_size, original_size=original_image_size
                )

            return rpn_output, det_output