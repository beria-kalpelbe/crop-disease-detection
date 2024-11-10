import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models

class SqueezeNetObjectDetectionWithConfidence(nn.Module):
    def __init__(self, num_classes=23, num_boxes=5):
        super(SqueezeNetObjectDetectionWithConfidence, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        # Load pretrained SqueezeNet and keep only the feature layers
        squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(squeezenet.features.children()))
        # freeze weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Bounding box regression head
        # Predicting 4 coordinates for each anchor point
        self.bbox_head = nn.Conv2d(in_channels=512, out_channels=num_boxes * 4, kernel_size=1)
        # Classification head
        # Predicting `num_classes` probabilities for each anchor point
        self.class_head = nn.Conv2d(in_channels=512, out_channels=num_boxes * num_classes, kernel_size=1)
        # Confidence head
        # Predicting 1 confidence score for each bounding box
        self.confidence_head = nn.Conv2d(in_channels=512, out_channels=num_boxes, kernel_size=1)
        # Activation functions for output heads
        self.softmax = nn.Softmax(dim=-1)  # for classification head
        self.sigmoid = nn.Sigmoid()        # for bounding box and confidence outputs (if normalized outputs are needed)

    def forward(self, x):
        # Step 1: Extract features using SqueezeNet backbone
        features = self.feature_extractor(x)  # Output shape: (batch_size, 512, H_f, W_f)
        # Step 2: Predict bounding boxes
        bbox_preds = self.bbox_head(features)  # Output shape: (batch_size, num_boxes*4, H_f, W_f)
        # Reshape to (batch_size, num_boxes, 4)
        bbox_preds = bbox_preds.view(x.size(0), self.num_boxes, 4, -1)  # Reshape to (batch_size, num_boxes, 4, H*W)
        bbox_preds = bbox_preds.permute(0, 3, 1, 2).contiguous()  # (batch_size, H*W, num_boxes, 4)
        bbox_preds = bbox_preds.view(x.size(0), -1, 4)  # Final shape: (batch_size, H*W*num_boxes, 4)
        # Ensure only the first num_boxes are kept
        bbox_preds = bbox_preds[:, :self.num_boxes, :]
        bbox_preds = self.sigmoid(bbox_preds)  # Normalize bounding box coordinates
        # Step 3: Predict class probabilities
        class_preds = self.class_head(features)  # Output shape: (batch_size, num_boxes*num_classes, H_f, W_f)
        # Reshape to (batch_size, num_boxes, num_classes)
        class_preds = class_preds.view(x.size(0), self.num_boxes, self.num_classes, -1)  # Reshape to (batch_size, num_boxes, num_classes, H*W)
        class_preds = class_preds.permute(0, 3, 1, 2).contiguous()  # (batch_size, H*W, num_boxes, num_classes)
        class_preds = class_preds.view(x.size(0), -1, self.num_classes)  # Final shape: (batch_size, H*W*num_boxes, num_classes)
        # Ensure only the first num_boxes are kept
        class_preds = class_preds[:, :self.num_boxes, :]
        class_preds = self.softmax(class_preds)  # Apply softmax
        # class_preds = torch.argmax(class_preds, dim=2)
        # Step 4: Predict confidence scores
        confidence_preds = self.confidence_head(features)  # Output shape: (batch_size, num_boxes, H_f, W_f)
        confidence_preds = confidence_preds.view(x.size(0), self.num_boxes, -1)  # Reshape to (batch_size, num_boxes, H*W)
        confidence_preds = confidence_preds.permute(0, 2, 1).contiguous()  # (batch_size, H*W, num_boxes)
        confidence_preds = confidence_preds.view(x.size(0), -1, 1)  # Final shape: (batch_size, H*W*num_boxes, 1)
        # Ensure only the first num_boxes are kept
        confidence_preds = confidence_preds[:, :self.num_boxes, :]
        confidence_preds = self.sigmoid(confidence_preds)  # Normalize confidence scores
        # remove the last dimension of confidence_preds
        confidence_preds = confidence_preds.squeeze(-1)
        return bbox_preds, class_preds.float(), confidence_preds


class SqueezeNetObjectDetectionWithFPN(nn.Module):
    def __init__(self, num_classes=23, num_boxes=5):
        super(SqueezeNetObjectDetectionWithFPN, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        # Load pretrained SqueezeNet backbone
        squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(squeezenet.features.children()))
        # freeze weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # FPN layers for multi-scale feature extraction
        self.fpn1 = nn.Conv2d(512, 256, kernel_size=1)
        self.fpn2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fpn4 = nn.Conv2d(512, 512, kernel_size=1)
        # Squeeze-and-Excitation (SE) block for attention
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.Sigmoid()
        )
        # Bounding box regression head with additional intermediate layers
        self.bbox_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_boxes * 4, kernel_size=1)
        )
        # Classification head with intermediate layers
        self.class_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_boxes * num_classes, kernel_size=1)
        )
        # Confidence head with intermediate layers
        self.confidence_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_boxes, kernel_size=1)
        )
        # Activation functions for output heads
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Step 1: Extract initial features using SqueezeNet backbone
        features = self.feature_extractor(x)
        # Step 2: Multi-scale feature refinement with FPN
        fpn_features = self.fpn1(features)
        fpn_features = self.fpn2(fpn_features)
        fpn_features = self.fpn3(fpn_features)
        fpn_features = self.fpn4(fpn_features)
        # Step 3: Apply SE attention
        se_weights = self.se_block(fpn_features)
        features = fpn_features * se_weights
        # Step 4: Predict bounding boxes with refinement
        bbox_preds = self.bbox_head(features)
        bbox_preds = bbox_preds.view(x.size(0), self.num_boxes, 4, -1)  # Reshape for multi-box predictions
        bbox_preds = bbox_preds.permute(0, 3, 1, 2).contiguous()
        bbox_preds = bbox_preds.view(x.size(0), -1, 4)  # Final shape: (batch_size, H*W*num_boxes, 4)
        bbox_preds = bbox_preds[:, :self.num_boxes, :]
        bbox_preds = self.sigmoid(bbox_preds)  # Normalize bounding box coordinates
        # Step 5: Predict class probabilities with refinement
        class_preds = self.class_head(features)
        class_preds = class_preds.view(x.size(0), self.num_boxes, self.num_classes, -1)
        class_preds = class_preds.permute(0, 3, 1, 2).contiguous()
        class_preds = class_preds.view(x.size(0), -1, self.num_classes)
        class_preds = class_preds[:, :self.num_boxes, :]
        class_preds = self.softmax(class_preds)  # Apply softmax over class dimension
        # Step 6: Predict confidence scores with refinement
        confidence_preds = self.confidence_head(features)
        confidence_preds = confidence_preds.view(x.size(0), self.num_boxes, -1)
        confidence_preds = confidence_preds.permute(0, 2, 1).contiguous()
        confidence_preds = confidence_preds.view(x.size(0), -1, 1)  # Final shape: (batch_size, H*W*num_boxes, 1)
        confidence_preds = confidence_preds[:, :self.num_boxes, :]
        confidence_preds = self.sigmoid(confidence_preds).squeeze(-1)  # Normalize and remove last dimension
        return bbox_preds, class_preds.float(), confidence_preds
