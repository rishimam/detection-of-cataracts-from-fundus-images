import torch
import torch.nn as nn
import torchvision.models as models


class CataractClassifier(nn.Module):
    
    def __init__(self, num_classes: int = 2, num_features: int = 14, use_features: bool = True, freeze_backbone: bool = True, dropout: float = 0.5):

        super(CataractClassifier, self).__init__()
        
        self.use_features = use_features
        
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        
        if freeze_backbone:
            for param in self.features[:-4].parameters():
                param.requires_grad = False
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        vgg_feature_size = 512 * 7 * 7  # 25088
        if use_features:
            combined_size = vgg_feature_size + num_features
        else:
            combined_size = vgg_feature_size
        
        # custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, handcrafted_features=None):

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # handmade features if provided
        if self.use_features and handcrafted_features is not None:
            x = torch.cat([x, handcrafted_features], dim=1)
        
        x = self.classifier(x)
        return x
    
    def unfreeze_backbone(self, num_layers: int = None):
        if num_layers is None:
            for param in self.features.parameters():
                param.requires_grad = True
        else:
            layers = list(self.features.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True