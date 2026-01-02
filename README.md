# ML Classifier — Naive Bayes Text Classification (C++)

A C++17 command-line program that trains a **text classifier** on labeled Piazza posts and predicts the most likely label for new posts using a **bag-of-words Naive Bayes model**.  
The project emphasizes **NLP fundamentals**, **probabilistic modeling**, and efficient use of C++ **container ADTs** (`map`, `set`) to build a complete end-to-end application.

---

## Overview

This classifier learns from previously labeled posts (training set) and predicts labels for unseen posts (test set).  
It treats each post as a **set of unique words** (bag-of-words model) and selects the label that maximizes a **log-probability score**.

Supported label sets are **data-driven** (no hardcoded classes): the model uses exactly the labels present in the training data.

---

## Key Features

- **Supervised text classification** using a Bernoulli-style Naive Bayes model
- Bag-of-words representation using **unique tokens** (duplicates ignored)
- Uses **log probabilities** for numerical stability
- **Smoothing / fallback handling** for unseen words and unseen label-word pairs
- Deterministic scoring by processing tokens in **alphabetical order**
- Command-line interface with:
  - Train-only mode (prints learned parameters)
  - Train + test mode (prints predictions and accuracy)
- Efficient implementation using `std::map` and `std::set`

---

## How It Works

During training, the program computes:

- Total number of training documents
- Vocabulary size (unique words)
- Per-label document counts (priors)
- Word frequencies overall and per-label

During prediction, for each candidate label, it computes:

- `log P(label)` + sum over words in post of `log P(word | label)`
- Predicts the label with the highest score (ties broken alphabetically)

---

## Project Structure

```
.
├── classifier.cpp          # Training + prediction implementation
├── csvstream.hpp           # CSV parsing helper
├── *.csv                   # Training/testing datasets
├── *.out.correct           # Reference outputs for verification
├── Makefile
```

---

## Build & Run

### Build
```bash
make -j4
```

### Train-only mode
```bash
./classifier.exe TRAIN_FILE
```

Example:
```bash
./classifier.exe train_small.csv
```

### Train + test mode
```bash
./classifier.exe TRAIN_FILE TEST_FILE
```

Example:
```bash
./classifier.exe train_small.csv test_small.csv
```

---

## Verify Output (Recommended)

Use `diff` to compare against provided reference outputs:

```bash
./classifier.exe train_small.csv > train_only.out
diff -y -B train_only.out train_small_train_only.out.correct | less
```

```bash
./classifier.exe train_small.csv test_small.csv > test.out
diff -y -B test.out test_small.out.correct | less
```

---

## Example Output (What You’ll See)

- Training summary (number of examples, vocabulary size)
- In train-only mode: learned priors and word likelihoods per label
- In test mode: per-post predictions + final accuracy, e.g.

```
performance: 245 / 332 posts predicted correctly
```

---

This project demonstrates practical fundamentals used in real ML/NLP systems:

- Feature extraction (tokenization + bag-of-words)
- Probabilistic modeling and smoothing
- Scalable counting with hash/map-based data structures
- Building a complete CLI ML pipeline: train → predict → evaluate
- Careful handling of edge cases and deterministic output


