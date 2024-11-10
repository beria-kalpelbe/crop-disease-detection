import torch
import torch.nn as nn

import torch

def ciou_loss(pred_boxes, target_boxes):
    """
    Complete IoU (CIoU) Loss.

    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes (N, 4) format [x1, y1, x2, y2]
        target_boxes (torch.Tensor): Ground truth bounding boxes (N, 4) format [x1, y1, x2, y2]

    Returns:
        torch.Tensor: CIoU loss value
    """
    # IoU Computation
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    # Area of intersection
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    # Area of predicted and target boxes
    pred_box_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_box_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    # Union area
    union = pred_box_area + target_box_area - intersection
    # IoU calculation
    iou = intersection / (union + 1e-6)
    # Center distance (L2 norm)
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    center_distance = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
    # Diagonal of the smallest enclosing box
    xc1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    yc1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    xc2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    yc2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    enclosing_diagonal = (xc2 - xc1) ** 2 + (yc2 - yc1) ** 2
    # Aspect ratio term (used for shape alignment)
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]
    # Aspect ratio consistency term (v)
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(target_w / (target_h + 1e-6)) - torch.atan(pred_w / (pred_h + 1e-6)), 2)
    # Alpha term (to weigh the aspect ratio loss)
    alpha = v / (1 - iou + v + 1e-6)
    # CIoU calculation
    ciou = iou - (center_distance / (enclosing_diagonal + 1e-6)) - alpha * v
    return 1 - ciou.mean()

def smooth_l1_loss(pred_bboxes, target_bboxes, beta=1.0):
    diff = torch.abs(pred_bboxes - target_bboxes)
    loss = torch.where(diff < beta, 0.5 * diff ** 2, diff - 0.5)
    return loss.mean()

class ObjectDetectionLoss(nn.Module):
    def __init__(self, num_classes, num_boxes):
        super(ObjectDetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.classification_loss = nn.CrossEntropyLoss()
        self.bbox_loss = smooth_l1_loss
        self.confidence_loss = nn.BCELoss()  # Use BCEWithLogitsLoss if confidence_preds are logits
    def forward(self, bbox_preds, class_preds, confidence_preds, bbox_targets, class_targets, confidence_targets):
        # Calculate the bounding box loss
        bbox_loss = self.bbox_loss(bbox_preds, bbox_targets)
        # Calculate the classification loss
        classification_loss = self.classification_loss(class_preds, class_targets.float())
        # Calculate the confidence loss
        confidence_loss = self.confidence_loss(confidence_preds, confidence_targets.float())
        # Combine the losses with optional weighting
        total_loss = 0.9*bbox_loss+ 0.05*classification_loss + 0.05*confidence_loss
        return total_loss