# Detection of Cataracts from Fundus Images

## Project Structure

```
cataract-classification/
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── preprocessing.py    # Image preprocessing & additional feature extraction
│   ├── dataset.py          # Dataset class
│   ├── model.py            # VGG16 classifier 
│   ├── data_pipeline.py    # Data loading pipeline
│   ├── train.py            # Training 
│   ├── evaluate.py         # Evaluation 
│   └── utils.py            # Utility functions
│
├── scripts/
│   └── main.py             # Main entry point with CLI
│
├── data/                   # Your dataset goes here
│   ├── 1_normal/
│   ├── 2_cataract/
│   ├── 3_retina/           # diabetic retinopathy images
│   └── 4_glaucoma/
│
└── results/                # Outputs
    ├── checkpoints/
    │   ├── best_model.pth
    │   └── final_model.pth
    ├── results/
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   └── metrics.json
    ├── training_history.png
    └── config.json
```

## Data

Retrieved from https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/code

## Installation

### 1. Clone or Download

```bash
# If using git
git clone <repo-url>
cd cataract-classification

# Or just download and extract the folder
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n cataract python=3.8
conda activate cataract

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Train with default settings
python scripts/main.py \
    --data-dir ./data \
    --output-dir ./results \
    --num-epochs 30

# Train without handcrafted features
python scripts/main.py \
    --data-dir ./data \
    --output-dir ./results \
    --no-features

# Evaluate existing model
python scripts/main.py \
    --data-dir ./data \
    --output-dir ./results \
    --eval-only \
    --model-path ./results/checkpoints/best_model.pth
```

### Advanced Usage

```bash
# Full control over training
python scripts/main.py \
    --data-dir ./data \
    --output-dir ./results \
    --batch-size 32 \
    --num-epochs 50 \
    --lr 0.001 \
    --dropout 0.5 \
    --early-stopping-patience 10 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --num-workers 4 \
    --verbose
```

## Command-Line Arguments

### Required Arguments

- `--data-dir`: Path to dataset directory

### Training Arguments

- `--output-dir`: Output directory (default: `./results`)
- `--batch-size`: Batch size (default: 32)
- `--num-epochs`: Number of epochs (default: 30)
- `--lr`: Learning rate (default: 0.001)
- `--early-stopping-patience`: Early stopping patience (default: 7)

### Model Arguments

- `--no-features`: Train without handcrafted features
- `--dropout`: Dropout rate (default: 0.5)
- `--unfreeze-backbone`: Train entire VGG16 (default: frozen except last block)

### Data Arguments

- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)

### Evaluation Arguments

- `--eval-only`: Skip training, only evaluate
- `--model-path`: Path to pre-trained model checkpoint

### Other Arguments

- `--seed`: Random seed (default: 42)
- `--num-workers`: Data loading workers (default: 2)
- `--verbose`: Print verbose output

## Dataset Setup

Your dataset should be organized as:

```
data/
├── 1_normal/       (or *_normal/)
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── 2_cataract/     (or *_cataract/)
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── 3_retina/       (optional, treated as normal)
    ├── image001.jpg
    └── ...
```

The code automatically detects folders ending in `_normal`, `_cataract`, or `_retina`, or `_glaucoma`.

## Expected Output

After training, you'll find:

### In `results/checkpoints/`:
- `best_model.pth` - Model with best validation loss
- `final_model.pth` - Model after training completes

### In `results/results/`:
- `confusion_matrix.png` - Test set confusion matrix
- `roc_curve.png` - ROC curve with AUC score
- `metrics.json` - All evaluation metrics

### In `results/`:
- `training_history.png` - Loss and accuracy curves
- `config.json` - Complete configuration used
