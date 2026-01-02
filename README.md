**ML Classifier**

**Description:**
This repository contains a compact implementation of a Multi-Variate Bernoulli Naive Bayes classifier written in C++ for classifying Piazza posts. The classifier learns which words are associated with each label from a training CSV file and then predicts labels for unseen posts. The code is intentionally simple and follows the EECS 280 project specification exactly so it can be used for automated grading.

Key points:
- Input: CSV files where the label is in the `tag` column and the post text is in the `content` column. Other columns are ignored.
- Model: Bag-of-words (presence/absence only). Each post is treated as a set of unique words; duplicate words within a post are ignored.
- Algorithm: Multi-Variate Bernoulli Naive Bayes using natural logarithms for probability sums.
- Smoothing rules (implemented per spec):
	- If a word occurs with a label in the training data: use P(w | label) = #(label, w) / #(label)
	- Else if the word occurs somewhere in training data: use P(w | label) = #(w) / #(total_posts)
	- Else (word never seen in training): use P(w | label) = 1 / #(total_posts)
- Tie-breaking: If two labels have the same log-probability score for a post, the classifier picks the label that is alphabetically first.

**Files:** `classifier.cpp`, `csvstream.hpp`, sample CSV datasets, and instructor `.correct` output files.

**Build**
- **Requirements:**: A C++17-capable compiler (e.g. `g++`) and `make`.
- **Build command:**

```
make classifier.exe
```

**Usage**
- Train-only mode:

```
./classifier.exe TRAIN_FILE
```

- Train + test mode:

```
./classifier.exe TRAIN_FILE TEST_FILE
```

- Exact usage message printed on bad args:

```
Usage: classifier.exe TRAIN_FILE [TEST_FILE]
```

**What the program prints**
- In train-only mode, the program prints per-training-post information (if any), the number of training examples, vocabulary size, a list of classes with counts and log-priors, and classifier parameters (label:word counts and log-likelihoods).
- In test mode, the program prints the number of training examples, then for each test post prints the correct label, predicted label, log-probability score, post content, and finally a performance summary (correct / total). The program sets `cout.precision(3)`.

**CSV format**
- The program uses the provided `csvstream.hpp` to read CSV files. Only the `tag` and `content` columns are used; other columns are ignored. You may assume content and tags are lowercase and contain no punctuation.

**Testing**
- Run the included test suite (compares produced output to instructor `.correct` files):

```
make test
```

**Notes**
- The classifier implements smoothing per spec and breaks ties alphabetically. Error messages and exit codes follow the project specification exactly (prints usage on wrong args; prints `Error opening file: <filename>` and returns non-zero when a file cannot be opened).
- All top-level code is in `classifier.cpp` as required.


