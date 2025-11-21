import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.dataset import CataractDataset
from src.utils import load_dataset, split_dataset
from src.config import Config


def get_transforms(config: Config, is_training: bool = False):

    if is_training:
        # augmentation for training
        return transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # val/test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(config: Config):
 
    all_paths, all_labels = load_dataset(config.data_dir)
    
    normal_count = all_labels.count(0)
    cataract_count = all_labels.count(1)
    
    print(f"Found {len(all_paths)} images:")
    print(f"  Normal: {normal_count}")
    print(f"  Cataract: {cataract_count}")
    
    # split
    (train_paths, train_labels), \
    (val_paths, val_labels), \
    (test_paths, test_labels) = split_dataset(
        all_paths, all_labels,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed
    )
    
    train_transform = get_transforms(config, is_training=True)
    val_test_transform = get_transforms(config, is_training=False)
    
    train_dataset = CataractDataset(
        train_paths, train_labels,
        transform=train_transform,
        extract_features=config.use_features
    )
    
    val_dataset = CataractDataset(
        val_paths, val_labels,
        transform=val_test_transform,
        extract_features=config.use_features
    )
    
    test_dataset = CataractDataset(
        test_paths, test_labels,
        transform=val_test_transform,
        extract_features=config.use_features
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    dataset_info = {
        'total_images': len(all_paths),
        'normal_count': normal_count,
        'cataract_count': cataract_count,
        'train_size': len(train_paths),
        'val_size': len(val_paths),
        'test_size': len(test_paths)
    }
    
    return train_loader, val_loader, test_loader, dataset_info