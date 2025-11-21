
import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, load_config, save_config
from src.data_pipeline import create_dataloaders
from src.model import CataractClassifier
from src.train import Train
from src.eval import Evaluate
from src.utils import set_seeds, get_device

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cataract Classification - Training and Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data args
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Path to save outputs')
    
    # Training args
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001,
                       dest='learning_rate',
                       help='Learning rate')
    parser.add_argument('--early-stopping-patience', type=int, default=7,
                       help='Early stopping patience (epochs)')
    
    # Model args
    parser.add_argument('--no-features', action='store_true',
                       help='Train without handcrafted features')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--freeze-backbone', action='store_true', default=True,
                       help='Freeze VGG16 backbone (except last block)')
    parser.add_argument('--unfreeze-backbone', dest='freeze_backbone', 
                       action='store_false',
                       help='Train entire VGG16 backbone')
    
    # Data split args
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio')
    
    # eval mode
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate pre-trained model (skip training)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pre-trained model checkpoint')
    
    # other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    
    # parse args
    args = parse_args()
    
    set_seeds(args.seed)
    
    # get device
    device = get_device()
    
    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_features=not args.no_features,
        early_stopping_patience=args.early_stopping_patience,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        device=str(device)
    )
    
    # output
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # print/save config
    print("CATARACT CLASSIFICATION - CONFIGURATION")
    config.print_config()
    save_config(config, Path(config.output_dir) / 'config.json')
    
    # data loaders
    print("LOADING DATA")
    
    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(config)
    
    print(f"\nDataset Statistics:")
    print(f"  Total images: {dataset_info['total_images']}")
    print(f"  Normal: {dataset_info['normal_count']}")
    print(f"  Cataract: {dataset_info['cataract_count']}")
    print(f"\nSplit:")
    print(f"  Train: {dataset_info['train_size']} images")
    print(f"  Val:   {dataset_info['val_size']} images")
    print(f"  Test:  {dataset_info['test_size']} images")
    print(f"\nBatches:")
    print(f"  Train: {len(train_loader)}")
    print(f"  Val:   {len(val_loader)}")
    print(f"  Test:  {len(test_loader)}")
    
    # handmade features
    if config.use_features:
        sample_batch = next(iter(train_loader))
        if len(sample_batch) == 3:  # images, features, labels
            _, sample_features, _ = sample_batch
            config.num_features = sample_features.shape[1]
            print(f"Detected {config.num_features} handcrafted features")
        else:
            print(" Warning: Expected 3-tuple batch but got different format")
            config.use_features = False
            print("Disabling feature fusion")
    
    # Create model
    print("MODEL ARCHITECTURE")
    
    model = CataractClassifier(
        num_classes=config.num_classes,
        num_features=config.num_features,
        use_features=config.use_features,
        freeze_backbone=config.freeze_backbone,
        dropout=config.dropout
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: VGG16-based Classifier")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Feature fusion:       {config.use_features}")
    print(f"  Dropout rate:         {config.dropout}")
    print(f"  Frozen backbone:      {config.freeze_backbone}")
    

    if not args.eval_only:
        # TRAINING MODE
        print("TRAINING")
        
        # Create trainer
        trainer = Train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        history = trainer.train()
        
        print("\n Training complete!")
        print(f"Best model saved to: {config.output_dir}/checkpoints/best_model.pth")
        
        # Plot training history
        print("\nGenerating training plots...")
        trainer.plot_history(save_path=Path(config.output_dir) / 'training_history.png')
        
        # Load best model for evaluation
        best_checkpoint_path = Path(config.output_dir) / 'checkpoints' / 'best_model.pth'
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"  Val Acc:  {checkpoint['val_acc']:.2f}%")
        
    else:

        print("EVALUATION MODE (SKIPPING TRAINING)")
        
        if args.model_path is None:
            raise ValueError(
                "Must provide --model-path for evaluation-only mode"
            )
        
        # Load pre-trained model
        print(f"\nLoading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_loss' in checkpoint:
            print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        if 'val_acc' in checkpoint:
            print(f"  Val Acc:  {checkpoint['val_acc']:.2f}%")
    

    print("EVALUATION ON TEST SET")
    
    # Create evaluator
    evaluator = Evaluate(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Evaluate
    metrics = evaluator.evaluate()
    
    # Print results
    print("TEST SET RESULTS")
    print(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Sensitivity: {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
    print(f"Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"AUC:         {metrics['auc']:.4f}")
    print("="*50)
    
    # Save results
    results_dir = Path(config.output_dir) / 'results'
    print(f"\n All results saved to: {results_dir}")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - metrics.json")
    if not args.eval_only:
        print("  - training_history.png")
    
    print("COMPLETE!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)