# Fake News Detection using Deep Learning

This project implements and evaluates various machine learning and deep learning models to classify news articles as fake or real, using the ISOT Fake News Dataset.

## Objective

Develop robust classifiers using traditional and deep learning models to detect fake news articles with high accuracy and reliability.

## Dataset

- Source: ISOT Fake News Dataset
- Real News: 21,417 articles from Reuters.com
- Fake News: 23,481 articles from unreliable sources flagged by Politifact and Wikipedia
- Fields: title, text, label, date
- Final dataset size after cleaning: 39,096 articles

## Preprocessing

1. Merged real and fake datasets with labels (0 = fake, 1 = real)
2. Removed:
   - Nulls and duplicates
   - Articles with 6 or fewer words (too short to provide meaningful context)
   - URLs, special characters, and extra whitespaces
   - Dataset-specific patterns (e.g., datelines like "WASHINGTON (Reuters) –" and image credits using regex)
3. Created two dataset variants:
   - With repeated header/footer patterns
   - Without repeated patterns
4. Tokenized and padded sequences for model training

## Models Explored

- Logistic Regression (Baseline)
- Bi-LSTM (Bidirectional Long Short-Term Memory)
- Bi-GRU (Bidirectional Gated Recurrent Unit)
- Transformer-based model using Multi-Head Attention

## Evaluation Metrics

- Accuracy
- Precision
- F1-score
- AUC (Area Under the Curve)
- EER (Equal Error Rate)

## Results Summary

| Model               | Accuracy | Precision | AUC   | EER   | F1-Score |
|--------------------|----------|-----------|-------|-------|----------|
| Logistic Regression| 0.9844   | 0.9800    | 0.9985| 0.0144| 0.98     |
| Bi-LSTM            | 0.9830   | 0.9902    | 0.9948| 0.0160| 0.98     |
| Bi-GRU             | 0.9884   | 0.9929    | 0.9994| 0.0108| 0.99     |
| Transformer        | 0.9884   | 0.9904    | 0.9992| 0.0094| 0.99     |

## Getting Started

### Clone the repository

```
git clone https://github.com/ishankkumar-007/FakeNewsDetection.git
cd FakeNewsDetection
```

### Create and activate the conda environment

```
conda env create --name tfgpu --file=env.yml
conda activate tfgpu
pip install ipykernel
python -m ipykernel install --user --name tfgpu --display-name "tfgpu (Python 3.12.3)"
```

### Run the notebooks

Navigate to the `src/final/` directory and run the desired notebook:

```
cd src/final/
```

## Directory Structure

```
.
├── docs
│   ├── FAKE NEWS DETECTION.pdf
│   ├── Fake_News_Detection_Report.pdf
│   └── ISOT_Fake_News_Dataset_ReadMe.pdf
├── env.yml
├── README.md
├── requirements.txt
└── src
    ├── final
    └── hyperparameter tuning notebooks

```

## Recommendation

- Use the Transformer model for a balanced performance across metrics.
- Use Bi-GRU when minimizing false positives is critical.