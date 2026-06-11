# Sequence to Sequence Date Conversion using Attention

A deep learning project that converts dates from multiple human readable formats into a standardized ISO format (`YYYY-MM-DD`) using Sequence to Sequence (Seq2Seq) models.

This project compares a **Vanilla Encoder Decoder LSTM** against a **Bahdanau Attention based Seq2Seq model**, evaluates performance, and visualizes attention weights.

---

## Problem Statement

Date formats vary widely across applications and regions.

Examples:

```text
"January 5, 2023" → "2023-01-05"
"07/04/1776" → "1776-04-07"
"25 JANUARY, 2050" → "2050-01-25"
"march 5, 3000" → "3000-03-05"
```

The goal of this project is to train a Seq2Seq model that learns how to normalize different date formats into a single standard representation.

---

## Objectives

* Build a **Vanilla Seq2Seq Encoder Decoder model**
* Build an **Attention based Seq2Seq model**
* Compare both architectures
* Evaluate exact match accuracy
* Visualize attention weights
* Analyze convergence speed and model behavior

---

## Dataset

A synthetic dataset was generated with:

* **300,000 date samples**
* Years ranging from **0001 to 3000**
* Multiple date formats
* Mixed casing support (`lower`, `UPPER`, `Title Case`)

### Supported Input Formats

```text
%B %d, %Y
%b %d, %Y
%d %B %Y
%d %b %Y
%m/%d/%Y
%d-%m-%Y
%Y/%m/%d
%d %B, %Y
%B %d %Y
%b %d %Y
%d/%m/%Y
```

Examples:

```text
January 05, 2023
JAN 18, 2040
march 5, 3000
07/04/1776
25 JANUARY, 2050
```

Target Format:

```text
YYYY-MM-DD
```

---

## Project Structure

```text
project-root/
│── data/
│   └── dataset.csv
│
│── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── seq2seq.py
│   ├── attention.py
│   ├── attn_decoder.py
│   └── attn_seq2seq.py
│
│── scripts/
│   ├── evaluate.py
│   └── visualize_attention.py
│
│── tests/
│   ├── test_dataset.py
│   ├── test_loader.py
│   └── test_model.py
│
│── train.py
│── train_attention.py
│── predict.py
│── dataset.py
│── dataloader.py
│── README.md
```

---

## Model Architectures

### 1. Vanilla Seq2Seq

Architecture:

```text
Input Sequence
      ↓
Encoder LSTM
      ↓
Context Vector
      ↓
Decoder LSTM
      ↓
Output Sequence
```

This model compresses the input into a fixed context vector and generates the target date character by character.

---

### 2. Attention Seq2Seq (Bahdanau Attention)

Architecture:

```text
Input Sequence
      ↓
Encoder LSTM
      ↓
Attention Mechanism
      ↓
Decoder LSTM
      ↓
Output Sequence
```

The decoder dynamically attends to important input regions while generating output characters.

---

## Training Details

### Hardware

* GPU: NVIDIA RTX 4060
* Framework: PyTorch

### Hyperparameters

```text
Embedding Dimension: 256
Hidden Size: 512
Batch Size: 64
Epochs: 15
Optimizer: Adam
Loss Function: CrossEntropyLoss
Teacher Forcing Ratio: 0.8
```

---

## Results

### Exact Match Accuracy

| Model             | Accuracy |
| ----------------- | -------- |
| Vanilla Seq2Seq   | 96.80%   |
| Attention Seq2Seq | 96.64%   |

### Observations

* Vanilla Seq2Seq slightly outperformed Attention.
* Attention did not significantly improve performance due to the short and structured nature of the task.
* Most prediction failures came from ambiguous numeric formats.

Example ambiguity:

```text
03/04/1438
```

Could mean:

```text
MM/DD/YYYY → March 4
```

or

```text
DD/MM/YYYY → 3 April
```

Since both patterns exist in the dataset, some ambiguity is unavoidable.

---

## Attention Visualization

Attention heatmaps were generated to understand how the model aligned input and output characters.

Examples:

```text
"march 5, 3000" → "3000-03-05"
"07/04/1776" → "1776-04-07"
"February 29, 2004" → "2004-02-29"
```

The visualizations show:

* Strong alignment for year copying
* Semantic mapping of month names to numeric months
* Correct localization of day values

---

## How to Run

### Training

Vanilla Model:

```bash
py train.py
```

Attention Model:

```bash
py train_attention.py
```

---

### Prediction (Inference)

Interactive Predictions:

```bash
py predict.py
```

Example:

```text
Enter date: march 5, 3000
Prediction: 3000-03-05
```

---

### Evaluation & Comparison

Run evaluation:

```bash
py scripts/evaluate.py
```

---

### Attention Visualization

Generate heatmaps:

```bash
py scripts/visualize_attention.py
```

---

### Tests

Test Dataset:

```bash
py tests/test_dataset.py
```

Test DataLoader:

```bash
py tests/test_loader.py
```

Test Model Forward Pass:

```bash
py tests/test_model.py
```

---

## Key Learnings

* Seq2Seq models work well for structured sequence transformation tasks.
* Attention mechanisms are not always beneficial for short sequences.
* Dataset ambiguity can become a limiting factor even with strong models.
* Attention visualization helps explain model behavior.

---

## Future Improvements

* Add Transformer based architecture
* Add beam search decoding
* Support multilingual date formats
* Remove ambiguous numeric date formats
* Build a web interface for live inference