import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List, Dict, Optional
from src.preprocessing import preprocess_image


class CataractDataset(Dataset):
  
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None, extract_features: bool = True):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.extract_features = extract_features
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        
        img_cropped, features = preprocess_image(img_path)
        
        if self.transform:
            img_cropped = self.transform(img_cropped)
        
        label = self.labels[idx]
        
        if self.extract_features: # Convert features to tensor
            feature_vector = torch.tensor(
                list(features.values()), 
                dtype=torch.float32
            )
            return img_cropped, feature_vector, label
        else:
            return img_cropped, label