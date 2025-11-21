import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Config:
    
    # Paths
    data_dir: str
    output_dir: str
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 0.001
    early_stopping_patience: int = 7
    
    # Model parameters
    num_classes: int = 2
    use_features: bool = True
    num_features: int = 14
    dropout: float = 0.5
    freeze_backbone: bool = True
    
    # Data parameters
    image_size: tuple = (224, 224)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Other
    num_workers: int = 2
    seed: int = 42
    device: str = 'cuda'
    
    def to_dict(self):
        config_dict = asdict(self)
        config_dict['image_size'] = list(self.image_size) # Convert tuple to list for JSON serialization
        return config_dict
    
    def print_config(self):
        for key, value in self.to_dict().items():
            print(f"  {key:25s}: {value}")


def load_config(config_path: Path) -> Config:
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    if 'image_size' in config_dict:     # Convert image_size back to tuple
        config_dict['image_size'] = tuple(config_dict['image_size'])
    
    return Config(**config_dict)


def save_config(config: Config, save_path: Path):
    with open(save_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
    print(f" Configuration saved to: {save_path}")