from tqdm import tqdm
import torch
import time
from datetime import timedelta
from wandb import AlertLevel

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            min_delta (float): Minimum change in monitored metric to be considered an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if there is an improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                self.early_stop = True

class Trainer:
    def __init__(self, model, optimizer, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.train_losses = []
        self.val_losses = []
        
    def _train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        for images, targets in tqdm(data_loader, desc="Training", leave=False):
            # Convert images to float32
            images = images.type(torch.float32).to(self.config.device)  # Changed to float32
            images = images.permute(0, 3, 1, 2)
            bbox_targets = targets['boxes'].to(self.config.device)
            # Ensure class_targets are of type long and within the valid range
            class_targets = targets['labels'].to(self.config.device)
            confidence_targets = targets['scores'].to(self.config.device)
            # Forward pass
            bbox_preds, class_preds, confidence_preds = self.model(images)
            # Calculate the loss
            loss = self.criterion(bbox_preds, class_preds, confidence_preds, bbox_targets, class_targets, confidence_targets)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def _validate_one_epoch(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Validating", leave=False):
                # Convert images to float32
                images = images.type(torch.float32).to(self.config.device)  # Changed to float32
                images = images.permute(0, 3, 1, 2)
                bbox_targets = targets['boxes'].to(self.config.device)
                class_targets = targets['labels'].to(self.config.device)
                confidence_targets = targets['scores'].to(self.config.device)
                # Forward pass
                bbox_preds, class_preds, confidence_preds = self.model(images)
                # Calculate the loss
                loss = self.criterion(bbox_preds, class_preds, confidence_preds, bbox_targets, class_targets, confidence_targets)
                total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def run(self,dataloader_train,dataloader_val):
        early_stopping = EarlyStopping(patience=10, min_delta=0.0)
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            train_loss = self._train_one_epoch(dataloader_train)
            val_loss = self._validate_one_epoch(dataloader_val)
            # Log the losses to wandb
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            epoch_time = (time.time() - start_time)/60
            print(f"Epoch [{epoch+1}/{self.config.num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {epoch_time:.2f}mins", end='\n')
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                wandb.alert(
                        title='Early Stoping Triggered',
                        text=f'Early stoping triggered at {epoch}th epoch with train loss={train_loss:.4f} and validation loss={val_loss:.4f}',
                        level=AlertLevel.WARN,
                        wait_duration=timedelta(minutes=0)
                        )
                break
        torch.save(self.model.state_dict(), "crop_disease_detection_model.pth")
        wandb.save("crop_disease_detection_model.pth")