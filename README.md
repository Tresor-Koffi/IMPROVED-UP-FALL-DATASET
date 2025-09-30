# Improved 3D Skeletons UP-Fall Dataset for Impact Fall Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/dataset-Zenodo-blue)](https://zenodo.org/records/12773013)

Official implementation and dataset for **"An Improved 3D Skeletons UP-Fall Dataset: Enhancing Data Quality for Efficient Impact Fall Detection"**

## 📋 Overview

This repository provides an enhanced version of the UP-Fall dataset with improved 3D skeleton data for accurate impact fall detection. We address critical limitations in the original dataset through advanced preprocessing techniques and manual validation.

### Key Features

- ✅ **High-Quality 3D Skeleton Data**: 33 keypoints extracted using MediaPipe BlazePose
- ✅ **Accurate Impact Labeling**: Manual and semi-automatic labeling of impact moments
- ✅ **Comprehensive Dataset**: 5 subjects × 5 fall scenarios + Activities of Daily Living
- ✅ **Validated Performance**: 99.5% accuracy with SVM, 98.5% with LSTM
- ✅ **Ready-to-Use Code**: Complete preprocessing and training pipeline

### What's Improved

| Original UP-Fall Dataset | Our Improved Dataset |
|-------------------------|---------------------|
| Mislabeled fall/non-fall events | Manually validated labels |
| Multiple subjects in frames | Isolated main subject |
| SMV threshold (1g) for falls | Impact-specific detection (2g threshold) |
| Mixed quality skeleton data | High-quality 3D skeleton extraction |
| Background noise interference | Clean, preprocessed frames |

## 📊 Dataset

### Download

The complete improved 3D skeleton UP-Fall dataset is available on Zenodo:
**[Download Dataset (Zenodo)](https://zenodo.org/records/12773013)**

### Dataset Statistics

- **Subjects**: 5 healthy young adults (9 males, 8 females, aged 18-24)
- **Activities**: 11 total (5 fall scenarios + 6 ADLs)
- **Skeleton Keypoints**: 33 joints per frame (x, y, z coordinates + visibility)
- **Cameras**: 2 views (frontal and lateral)
- **Frame Selection**: Only sequences with ≥100 frames included

### File Naming Convention

Files follow the pattern: `C{camera}S{subject}A{activity}T{trial}.csv`

- **C**: Camera (1 = frontal, 2 = lateral)
- **S**: Subject (1-5)
- **A**: Activity (1-11)
- **T**: Trial (1-3)

**Example**: `C1S1A1T1.csv` = Camera 1, Subject 1, Activity 1, Trial 1

### Activities

| ID | Category | Description | Duration | Abbreviation |
|----|----------|-------------|----------|--------------|
| 1 | FALL | Falling forward using hands | 10s | FH |
| 2 | FALL | Falling forward using knees | 10s | FF |
| 3 | FALL | Falling backward | 10s | FB |
| 4 | FALL | Falling sideward | 10s | FS |
| 5 | FALL | Falling attempting to sit | 10s | FE |
| 6 | ADL | Walking | 60s | W |
| 7 | ADL | Standing | 60s | S |
| 8 | ADL | Sitting | 60s | ST |
| 9 | ADL | Picking up object | 10s | P |
| 10 | ADL | Jumping | 30s | J |
| 11 | ADL | Laying | 60s | L |

### Data Format

Each CSV file contains:
- **Frame number**
- **33 3D keypoints** (x, y, z coordinates)
- **Visibility scores** for each keypoint
- **Impact label** (0 = no impact, 1 = impact detected)
- **Timestamp**

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum
- GPU recommended for deep learning models

### Setup

```bash
# Clone the repository
git clone https://github.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset.git
cd 3D_skeletons-UP-Fall-Dataset

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Quick Start

```python
import pandas as pd
import numpy as np
from src.models import SVMClassifier, LSTMClassifier

# Load skeleton data
data = pd.read_csv('data/subjects/subject1/C1S1A1T1.csv')
print(f"Loaded {len(data)} frames")
print(f"Shape: {data.shape}")

# Display first few rows
print(data.head())
```

### Data Loading Example

```python
from src.utils import DataLoader

# Initialize data loader
loader = DataLoader('data/subjects/')

# Load specific subject
subject1_data = loader.load_subject(1)

# Load specific activity
fall_data = loader.load_activity('fall')

# Split data for training
X_train, X_test, y_train, y_test = loader.train_test_split(test_size=0.2)
```

### Training Models

#### Machine Learning Models

```python
from src.models import train_ml_models

# Train multiple ML models
results = train_ml_models(
    data_path='data/subjects/',
    models=['SVM', 'RF', 'MLP', 'KNN'],
    test_size=0.2
)

# Display results
for model, metrics in results.items():
    print(f"{model}: Accuracy = {metrics['accuracy']:.2%}")
```

#### Deep Learning Models

```python
from src.models import LSTMClassifier

# Initialize and train LSTM
model = LSTMClassifier(
    input_dim=99,  # 33 keypoints × 3 coordinates
    hidden_dim=128,
    num_layers=2,
    num_classes=2
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

# Evaluate
accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

### Visualization

```python
from src.utils import Visualizer

# Create visualizer
viz = Visualizer()

# Visualize skeleton sequence
viz.plot_skeleton_sequence('data/subjects/subject1/C1S1A1T1.csv')

# Visualize fall detection over time
viz.plot_impact_detection('data/subjects/subject1/C1S1A1T1.csv')

# Create confusion matrix
viz.plot_confusion_matrix(y_true, y_pred, model_name='LSTM')
```

## 📈 Results

### Machine Learning Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM** | **99.50%** | **99.50%** | **99.50%** | **99.50%** |
| MLP | 99.47% | 99.70% | 97.83% | 98.74% |
| RF | 99.28% | 98.47% | 99.18% | 98.47% |
| KNN | 98.35% | 98.45% | 99.03% | 98.74% |
| NB | 98.85% | 97.57% | 97.94% | 97.76% |
| SGD | 94.47% | 97.71% | 95.85% | 96.77% |

### Deep Learning Performance

| Model | Accuracy | Precision | Sensitivity | Specificity |
|-------|----------|-----------|-------------|-------------|
| **LSTM** | **98.50%** | **97.83%** | **97.76%** | **100%** |
| BiLSTM | 94.00% | 95.89% | 100% | 81.82% |
| CNN | 93.00% | 95.27% | 100% | 78.79% |
| STGCN | 92.00% | 91.43% | 95.59% | 84.38% |

### Key Findings

- **SVM** achieved the highest overall accuracy for machine learning approaches
- **LSTM** showed the best performance among deep learning models with perfect specificity
- All models achieved >90% accuracy, demonstrating dataset quality
- Impact detection accuracy significantly improved compared to original dataset

## 📁 Repository Structure

```
3D_skeletons-UP-Fall-Dataset/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
│
├── data/
│   └── subjects/
│       ├── subject1/
│       ├── subject2/
│       ├── subject3/
│       ├── subject4/
│       └── subject5/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── image_processor.py
│   │   ├── skeleton_extractor.py
│   │   └── background_remover.py
│   ├── labeling/
│   │   ├── manual_labeler.py
│   │   └── semi_automatic_labeler.py
│   ├── models/
│   │   ├── ml_models.py
│   │   ├── dl_models.py
│   │   └── lstm_model.py
│   └── utils/
│       ├── data_loader.py
│       ├── visualizer.py
│       └── metrics.py
│
├── examples/
│   ├── quick_start.py
│   ├── train_svm.py
│   └── train_lstm.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_visualization.ipynb
│
├── docs/
│   ├── paper.pdf
│   ├── INSTALLATION.md
│   └── API_REFERENCE.md
│
└── tests/
    ├── test_data_loader.py
    └── test_models.py
```

## 🔬 Methodology

Our preprocessing pipeline includes:

1. **Image Cropping**: Crop to 500×700 pixels to isolate main subject
2. **Background Removal**: GrabCut algorithm to remove background noise
3. **3D Skeleton Extraction**: MediaPipe BlazePose for 33 keypoints
4. **Impact Labeling**: Semi-automatic detection using 2g SMV threshold
5. **Manual Validation**: Visual inspection and correction of labels

For detailed methodology, see our [paper](docs/paper.pdf).

## 📖 Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{koffi2024improved,
  title={An Improved 3D Skeletons UP-Fall Dataset: Enhancing Data Quality for Efficient Impact Fall Detection},
  author={Koffi, Tresor Y. and Mourchid, Youssef and Hindawi, Mohammed and Dupuis, Yohan},
  journal={Conference Proceedings},
  year={2024},
  organization={CESI LINEACT Laboratory},
  url={https://zenodo.org/records/12773013}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 👥 Authors

- **Tresor Y. Koffi** - CESI LINEACT Laboratory, Dijon, France
- **Youssef Mourchid** - CESI LINEACT Laboratory, Dijon, France
- **Mohammed Hindawi** - CESI LINEACT Laboratory, Lyon, France
- **Yohan Dupuis** - CESI LINEACT Laboratory, Paris La Defense, France

## 📧 Contact

For questions or collaborations, please open an issue or contact:
- Email: tresor.koffi@cesi.fr
- GitHub: [@Tresor-Koffi](https://github.com/Tresor-Koffi)

## 🙏 Acknowledgments

- Original UP-Fall dataset creators
- CESI LINEACT Laboratory, France
- MediaPipe team for BlazePose algorithm
- Research community for valuable feedback

## 📊 Related Resources

- [Original UP-Fall Dataset](https://sites.google.com/up.edu.mx/har-up/)
- [MediaPipe BlazePose](https://google.github.io/mediapipe/solutions/pose.html)
- [Zenodo Dataset Repository](https://zenodo.org/records/12773013)

---

**Note**: The complete dataset with all preprocessing code is available. For the raw UP-Fall images, please refer to the original UP-Fall dataset repository.
