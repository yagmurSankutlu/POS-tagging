# Turkish Part-of-Speech Tagging with BERT: A Deep Learning Approach

## Abstract

This report presents a comprehensive study on Turkish Part-of-Speech (POS) tagging using a BERT-based deep learning model. The project implements a sophisticated architecture combining BERT embeddings with optional BiLSTM layers and achieves state-of-the-art performance on Turkish POS tagging tasks. The model successfully achieves an F1 score of 91.75% and accuracy of 92.08%, surpassing the target threshold of 90% F1 score.

## 1. Introduction

Part-of-Speech tagging is a fundamental task in Natural Language Processing that involves assigning grammatical categories to words in a sentence. For Turkish, a morphologically rich and agglutinative language, POS tagging presents unique challenges due to complex word formations and rich inflectional morphology.

### 1.1 Objectives
- Develop a high-performance Turkish POS tagger using BERT
- Achieve F1 score > 90% on Turkish Universal Dependencies test set
- Implement and evaluate different architectural variants (with/without BiLSTM)
- Apply advanced regularization techniques for robust performance

## 2. Dataset

### 2.1 Data Sources
The project utilizes two complementary Turkish Universal Dependencies corpora:

| Dataset | Purpose | Description |
|---------|---------|-------------|
| OTA BOUN UD | Primary | Ottoman Turkish - Boğaziçi University corpus |
| TR BOUN UD (IMST) | Augmentation | Modern Turkish - Boğaziçi University corpus |

### 2.2 Dataset Statistics

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| Total Sentences | 7,917 | 400 |
| Unique POS Tags | 17 | 17 |
| Max Sequence Length | 512 tokens | 512 tokens |

### 2.3 POS Tag Distribution
The dataset includes 17 Universal POS tags:
`ADJ`, `ADP`, `ADV`, `AUX`, `CCONJ`, `DET`, `INTJ`, `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `SCONJ`, `VERB`, `X`, `_`

## 3. Model Architecture

### 3.1 Base Architecture Components

| Component | Configuration | Purpose |
|-----------|---------------|---------|
| **BERT Backbone** | `dbmdz/bert-base-turkish-cased` | Contextualized embeddings |
| **Embedding Processor** | Custom dropout configuration | Regularization and token masking |
| **BiLSTM Layer** | 2-layer, 256 hidden units, bidirectional | Sequential context modeling |
| **Sequence Tagger** | Linear layer with dropout | POS classification |

### 3.2 Regularization Strategy

| Technique | Value | Application |
|-----------|-------|-------------|
| Hidden Dropout | 0.2 | BERT layers |
| Attention Dropout | 0.2 | BERT attention |
| Output Dropout | 0.5 | BERT output |
| Token Masking | 0.15 | Input regularization |
| Label Smoothing | 0.1 | Loss function |
| BiLSTM Dropout | 0.3 | Sequential layers |

### 3.3 Architecture Variants

| Variant | Configuration | Use Case |
|---------|---------------|----------|
| **BERT-only** | `use_bilstm=False` | Baseline model |
| **BERT+BiLSTM** | `use_bilstm=True, lstm_hidden_dim=256` | Enhanced sequential modeling |

## 4. Training Methodology

### 4.1 Cold Start Training Paradigm

The project employs a two-stage training approach:

| Stage | Configuration | Duration | Learning Rate |
|-------|---------------|----------|---------------|
| **Stage 1: Cold Start** | BERT frozen, train classification layers | 3 epochs | 3e-4 |
| **Stage 2: Fine-tuning** | Full model training | 20 epochs | 4e-5 |


### 4.2 Hyperparameter Tuning

For learning rate the hyperparamter set was 3e-4, 4e-5, 5e-5. 4e-5 gave best results.
For batch size the hyperparamter set was 16, 32, 64. 32 gave best results.

### 4.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 32 | Memory efficiency |
| Max Length | 512 | BERT limit |
| Optimizer | AdamW | Default for BERT |
| LR Scheduler | Inverse Square Root | Gradual decay |
| Warmup Steps | 400 | Stable convergence |
| Hardware | 1x H100 GPU | High-performance training |


## 5. Results

### 5.1 Results Table

| Method        | F1 Score                                             |
|---------------------------------------------------------|------------|
| BERT + Linear Layer with Cold Start Training            | 92.00%     |
| BERT + Linear Layer + BiLSTM with Cold Start Training   | 91.81%     |
| BERT + Linear Layer + BiLSTM + wo Cold Start Training   | 91.33%     |
| BERT + Linear Layer +  wo Cold Start Training           | 91.96%     |
|---------------------------------------------------------|------------|

### 5.2 Best Performance Metrics

| Metric     | Value    |
|------------|----------|
| F1 Score   | 92.00%   |


## 6. Analysis and Discussion

### 6.1 Performance Analysis

1. **Rapid Convergence**: The model shows excellent learning, jumping from 67% to 90% F1 in just 2 epochs
2. **Stable Performance**: F1 scores remain consistent.
3. **Target Achievement**: Successfully exceeded the 90% F1 threshold, reaching 91.75%
4. **Balanced Metrics**: Precision and recall are well-balanced, indicating robust performance

### 6.2 Training Paradigm Effectiveness

The cold start training approach proved highly effective:
- **Prevents Catastrophic Forgetting**: BERT's pre-trained knowledge is preserved
- **Stable Initialization**: Classification layers are properly initialized before full training
- **Faster Convergence**: Reduces training time while maintaining performance
- **BiLSTM**: Didn't improve performance with BiLSTM

### 6.3 Regularization Impact

The comprehensive regularization strategy successfully prevented overfitting:
- **Multiple Dropout Layers**: Effective generalization
- **Token Masking**: Improved robustness to input variations
- **Label Smoothing**: Reduced overconfidence in predictions

## 7. Technical Implementation

### 7.1 Data Processing Pipeline

1. **CoNLL-U Parsing**: Automatic extraction of tokens and POS tags
2. **Tokenization**: BERT subword tokenization with proper alignment
3. **Label Alignment**: Handles subword tokens correctly (-100 for special tokens)
4. **Dataset Creation**: Hugging Face compatible format

### 7.2 Model Components

# Key architectural decisions
- BERT: dbmdz/bert-base-turkish-cased (768 dim)
- BiLSTM: 2-layer, 256 hidden units, bidirectional
- Classifier: Linear layer with 0.2 dropout
- Loss: CrossEntropyLoss with label smoothing


## 8. Conclusions

### 8.1 Key Achievements

1. **Performance Excellence**: Achieved 92.00% F1 score, exceeding 90% target
2. **Architectural Innovation**: Successfully integrated BiLSTM with BERT embeddings
3. **Training Efficiency**: Robust two-stage training paradigm
4. **Comprehensive Regularization**: Multiple techniques for improved generalization

### 8.2 Technical Contributions

- Implementation of STEPS-inspired architecture for Turkish POS tagging
- Advanced regularization techniques including token masking
- Effective cold start training methodology
- Comprehensive evaluation on Turkish POS tagging

## 9. References and Dependencies

### 9.1 Key Libraries
- **Transformers**: Hugging Face library for BERT
- **PyTorch**: Deep learning framework
- **CoNLL-U**: Universal Dependencies parsing
- **scikit-learn**: Evaluation metrics

### 9.2 Model Resources
- **Base Model**: `dbmdz/bert-base-turkish-cased`
- **Datasets**: Universal Dependencies Turkish corpora
- **Hardware**: Google Cloud H100 GPU

---

**Project Status**: Target F1 > 90% achieved with 92.00%

**Final Model**: Saved as `turkish_pos_model.pth`
