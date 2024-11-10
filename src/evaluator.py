import cv2
import numpy as np
import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, config, X_val) -> None:
        self.model = model
        self.config = config
        self.predictions = []
        self.X_val = X_val
        self.mAP = None
    
    def _load_test_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        return image

    def _test_one_image(self,image_path):
        image = self._load_test_image(image_path)
        image = torch.tensor(image)
        self.model.eval()
        with torch.no_grad():
            image = image.type(torch.float32).to(self.config.device)
            image = image.permute(2, 0, 1).unsqueeze(0)
            bbox_preds, class_preds, confidence_preds = self.model(image)
            class_preds = torch.argmax(class_preds, dim=2)
            return bbox_preds, class_preds, confidence_preds

    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes.
        Boxes are in the format: [x1, y1, x2, y2]
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        # Calculate intersection area
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        # Calculate areas of each box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        # Calculate IoU
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou
    
    def _calculate_average_precision(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """
        Calculate Average Precision (AP) for a single class.
        pred_boxes: List of predicted boxes with confidence scores: [(x1, y1, x2, y2, score), ...]
        gt_boxes: List of ground truth boxes: [(x1, y1, x2, y2), ...]
        """
        # Sort predictions by confidence score in descending order
        pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
        # Track true positives (TP) and false positives (FP)
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        # Keep track of which GT boxes have been matched
        matched_gt_boxes = set()
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            # Check against each GT box
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt_boxes:
                    continue  # Skip if this GT box is already matched

                iou = self._calculate_iou(pred_box[:4], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            # Determine if this is a true positive or false positive
            if best_iou >= iou_threshold:
                tp[i] = 1  # Correct detection
                matched_gt_boxes.add(best_gt_idx)  # Mark GT box as matched
            else:
                fp[i] = 1  # Incorrect detection
        # Calculate cumulative sums for TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (len(gt_boxes) + 1e-6)
        # Calculate AP (Average Precision) as area under the precision-recall curve
        ap = 0.0
        for i in range(1, len(precision)):
            ap += (recall[i] - recall[i - 1]) * precision[i]
        return ap
    
    def _calculate_map(self, pred_boxes_dict, gt_boxes_dict, iou_threshold=0.5):
        """
        Calculate Mean Average Precision (mAP) at a given IoU threshold for all classes.
        pred_boxes_dict: Dictionary with class-wise predicted boxes, e.g., {class_id: [(x1, y1, x2, y2, score), ...]}
        gt_boxes_dict: Dictionary with class-wise ground truth boxes, e.g., {class_id: [(x1, y1, x2, y2), ...]}
        """
        ap_values = []
        for class_id in gt_boxes_dict.keys():
            pred_boxes = pred_boxes_dict.get(class_id, [])
            gt_boxes = gt_boxes_dict[class_id]
            ap = self._calculate_average_precision(pred_boxes, gt_boxes, iou_threshold)
            ap_values.append(ap)
        # mAP is the mean of AP values for each class
        mAP = np.mean(ap_values) if ap_values else 0.0
        return mAP
    
    def _get_shape_image(self,image_path):
        image = cv2.imread(image_path)
        return image.shape[0], image.shape[1]
    
    def _process_model_output(self, pred_boxes, pred_classes, pred_confidences, confidence_threshold=0.5):
        """
        Process model outputs to keep predictions above a confidence threshold.
        Returns a dictionary with class-wise predictions in the format:
        {class_id: [(x1, y1, x2, y2, confidence), ...]}
        """
        processed_predictions = {}
        for bbox, cls, conf in zip(pred_boxes, pred_classes, pred_confidences):
            if conf >= confidence_threshold:
                if cls not in processed_predictions:
                    processed_predictions[cls] = []
                processed_predictions[cls].append((*bbox, conf))
        return processed_predictions

    def _process_ground_truth(self, gt_boxes, gt_classes):
        """
        Process ground truth data into a dictionary format for mAP calculation.
        Returns a dictionary with class-wise ground truth boxes in the format:
        {class_id: [(x1, y1, x2, y2), ...]}
        """
        processed_gt = {}
        for bbox, cls in zip(gt_boxes, gt_classes):
            if cls not in processed_gt:
                processed_gt[cls] = []
            processed_gt[cls].append(bbox)
        return processed_gt
    
    def run_predictions(self):
        for i, image_path in tqdm(enumerate(self.X_val['image_path'].unique()), total=len(self.X_val['image_path'].unique()), desc="Running predictions"):
            shapex, shapey = self._get_shape_image(image_path)
            X_val_copy = self.X_val.copy()
            X_val_copy = X_val_copy[X_val_copy['image_path']==image_path]
            X_val_copy.loc[X_val_copy['image_path']==image_path, 'xmax'] = X_val_copy.loc[X_val_copy['image_path']==image_path, 'xmax']/shapex
            X_val_copy.loc[X_val_copy['image_path']==image_path, 'ymax'] = X_val_copy.loc[X_val_copy['image_path']==image_path, 'ymax']/shapey
            X_val_copy.loc[X_val_copy['image_path']==image_path, 'xmin'] = X_val_copy.loc[X_val_copy['image_path']==image_path, 'xmin']/shapex
            X_val_copy.loc[X_val_copy['image_path']==image_path, 'ymin'] = X_val_copy.loc[X_val_copy['image_path']==image_path, 'ymin']/shapey
            gt_boxes = X_val_copy[['xmin','ymin','xmax','ymax']].values.tolist()
            gt_classes = X_val_copy['class_id'].values.tolist()
            bbox, class_, confidence = self._test_one_image(image_path)
            confidence_sorted, indices = torch.sort(confidence, descending=True)
            bbox[0] = bbox[0][indices]
            class_[0] = class_[0][indices]
            pred = {
                    "pred_boxes": bbox[0].cpu().numpy()[:len(gt_classes)].tolist(),
                    "pred_classes": class_[0].cpu().numpy()[:len(gt_classes)].tolist(),
                    "pred_confidences": confidence_sorted[0].cpu().numpy()[:len(gt_classes)].tolist(),
                    "gt_boxes": gt_boxes,
                    "gt_classes": gt_classes
                }
            self.predictions.append(pred)
            
    def run_map_calculation(self):
        # Initialize dictionaries to accumulate predictions and ground truths for all images
        self.run_predictions()
        pred_boxes_dict = {}
        gt_boxes_dict = {}
        # Process each image's predictions and ground truth
        for data in self.predictions:
            # Process model predictions
            processed_predictions = self._process_model_output(
                data["pred_boxes"], data["pred_classes"], data["pred_confidences"], confidence_threshold=0.9
            )
            # Update pred_boxes_dict with processed predictions
            for cls_id, boxes in processed_predictions.items():
                if cls_id not in pred_boxes_dict:
                    pred_boxes_dict[cls_id] = []
                pred_boxes_dict[cls_id].extend(boxes)
            # Process ground truth boxes
            processed_gt = self._process_ground_truth(data["gt_boxes"], data["gt_classes"])
            # Update gt_boxes_dict with processed ground truths
            for cls_id, boxes in processed_gt.items():
                if cls_id not in gt_boxes_dict:
                    gt_boxes_dict[cls_id] = []
                gt_boxes_dict[cls_id].extend(boxes)

        # Calculate mAP at IoU threshold 0.5
        self.mAP = self._calculate_map(pred_boxes_dict, gt_boxes_dict, iou_threshold=0.5)
        print(f"mAP at IoU=0.5: {self.mAP}")
        