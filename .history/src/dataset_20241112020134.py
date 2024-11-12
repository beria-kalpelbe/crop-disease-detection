import cv2
import torch
from torch.utils.data import Dataset
import gdown
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import re
import os

class DatasetInitializer():
    def __init__(self, data_path:str='data'):
        self.DATA_DIR = data_path
        self._download_data()
        self._unzip_images()
        self.train, self.test, self.ss = self._read_csv_files()
        self.id_class_match = self._get_id_classes_correspondances()
        self.X_train, self.X_val = self._get_train_val_slpit()
        
    def _download_file_from_drive(self, url, filename):
        pat = re.compile('https://drive.google.com/file/d/(.*)') 
        g = re.match(pat,url)
        if g:
            id = g.group(1)
            down_url = f'https://drive.google.com/uc?id={id}'
            os.system(f'gdown {down_url} -O {os.path.join(self.DATA_DIR, filename)}')
            os.system(f'gdown {down_url}')
        else:
            print(f"Warning: Could not extract ID from URL: {url}")
        
    def _download_data(self):
        url_list = {
            "images.zip": "https://drive.google.com/file/d/1rY8pjQ27h9petBcZP0RxkYHoyu8RvToq",
            "Train.csv": "https://drive.google.com/file/d/1iToVuuZd48c-I232GWk0FEIzeUgEPRMw",
            "Test.csv""https://drive.google.com/file/d/1oJO8xR_i_FSkyePLsJ-Cd_1EaKn5OJxO",
            "https://drive.google.com/file/d/1Z3JX-5946GKskwwaOpU-KUw7lcR9EY7F"
        }
        pat = re.compile('https://drive.google.com/file/d/(.*)') 
        for i, url in enumerate(url_list):
            self._download_file_from_drive(url, filenames[i])
        
        gdown.download(, output=f"{self.DATA_DIR}/images.zip", quiet=False)
        gdown.download(, output=f"{self.DATA_DIR}/Train.csv", quiet=False)
        gdown.download(, output=f"{self.DATA_DIR}/Test.csv", quiet=False)
        gdown.download(, output=f"{self.DATA_DIR}/SampleSubmission.csv", quiet=False)
    
    def _unzip_images(self):
        shutil.unpack_archive(self.DATA_DIR+'/images.zip', 'images')
        
    def _read_csv_files(self):
        train = pd.read_csv(self.DATA_DIR / 'Train.csv')
        test = pd.read_csv(self.DATA_DIR / 'Test.csv')
        ss = pd.read_csv(self.DATA_DIR / 'SampleSubmission.csv')
        # Add an image_path column
        train['image_path'] = [Path('images/' + x) for x in train.Image_ID]
        test['image_path'] = [Path('images/' + x) for x in test.Image_ID]
        # Map str classes to ints (label encoding targets)
        class_mapper = {x:y for x,y in zip(sorted(train['class'].unique().tolist()), range(train['class'].nunique()))}
        train['class_id'] = train['class'].map(class_mapper)
        return train, test, ss
    
    def _get_id_classes_correspondances(self):
        id_class_match = {}
        for i in range(len(self.train)):
            row = self.train.iloc[i]
            if row['class_id'] not in id_class_match.keys():
                id_class_match[row['class_id']] = row['class']
        id_class_match = dict(sorted(id_class_match.items(), reverse=False))
        return id_class_match
    
    def _get_train_val_slpit(self):
        train_unique_imgs_df = self.train.drop_duplicates(subset = ['Image_ID'], ignore_index = True)
        X_train, X_val = train_test_split(train_unique_imgs_df, test_size = 0.2, stratify=train_unique_imgs_df['class'], random_state=42)
        X_train = self.train[self.train.Image_ID.isin(X_train.Image_ID)]
        X_val = self.train[self.train.Image_ID.isin(X_val.Image_ID)]
        return X_train, X_val
    

class CropDiseaseDataset(Dataset):
    def __init__(self, dataframe, num_boxes=5, num_classes=50):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image metadata and annotations.
            num_boxes (int): The fixed number of bounding boxes to output per image.
        """
        self.dataframe = dataframe
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.image_paths = self.dataframe['image_path'].unique()
        self.image_path_indices = {path: i for i, path in enumerate(self.image_paths)}
        self.DATA_DIR = './data'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the data for the current index
        image_path = self.image_paths[idx]
        # Using .loc to avoid SettingWithCopyWarning
        sub_dataframe = self.dataframe[self.dataframe['image_path'] == image_path].copy()
        # Load the image
        image = cv2.imread(image_path)
        # Normalize the bounding box coordinates using .loc
        sub_dataframe.loc[:, 'xmin'] = sub_dataframe['xmin'] / image.shape[1]
        sub_dataframe.loc[:, 'ymin'] = sub_dataframe['ymin'] / image.shape[0]
        sub_dataframe.loc[:, 'xmax'] = sub_dataframe['xmax'] / image.shape[1]
        sub_dataframe.loc[:, 'ymax'] = sub_dataframe['ymax'] / image.shape[0]
        # Create the bounding box tensor
        boxes = []
        classes = []
        confidences = []
        for i in range(self.num_boxes):
            if i < len(sub_dataframe):
                xmin = sub_dataframe.iloc[i]['xmin']
                ymin = sub_dataframe.iloc[i]['ymin']
                xmax = sub_dataframe.iloc[i]['xmax']
                ymax = sub_dataframe.iloc[i]['ymax']
                _class = sub_dataframe.iloc[i]['class_id']
                confidence = sub_dataframe.iloc[i]['confidence']
            else:
                xmin, ymin, xmax, ymax = 0, 0, 0, 0
                _class = 0
                confidence = 0
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(_class)
            confidences.append(confidence)
        bbox = torch.tensor(boxes, dtype=torch.float32)
        confidence = torch.tensor(confidences, dtype=torch.float32)
        class_id = torch.tensor(classes, dtype=torch.int64)
        one_hot_class = torch.nn.functional.one_hot(class_id, num_classes=self.num_classes)
        # Prepare the target
        target = {
            'boxes': bbox,  # Shape: (num_boxes, 4)
            'labels': one_hot_class,  # Shape: (num_boxes,23,)
            'scores': confidence  # Shape: (num_boxes,)
        }
        # Apply transformations
        image = cv2.resize(image, (224, 224))  # Resize to your model's input size
        image = image / 255.0  # Normalize to [0, 1]

        return image, target