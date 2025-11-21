import torch
import numpy as np
import random
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split


def set_seeds(seed: int = 42): # this makes things deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using device: {device}")
    return device


def load_dataset(root_dir: str):
    image_paths = []
    labels = []
    
    # Scan all subdirectories
    for img_path in glob.glob(str(Path(root_dir) / "*/**")):
        class_name = img_path.split("/")[-2].split("_")[-1]
        
        if class_name == "cataract":
            image_paths.append(img_path)
            labels.append(1)
        elif class_name in ["normal", "retina", "glaucoma"]:
            image_paths.append(img_path)
            labels.append(0)
    
    return image_paths, labels


def split_dataset(image_paths, labels, train_ratio=0.7, val_ratio=0.15, seed=42):
    test_ratio = 1 - train_ratio - val_ratio
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )
    
    # val vs test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=seed
    )
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params