**ML Classifier**

- **Description:**: A small Multi-Variate Bernoulli Naive Bayes classifier for Piazza posts. The program trains on a CSV file containing `tag` and `content` columns and optionally predicts labels for a test CSV.
- **Files:**: `classifier.cpp`, `csvstream.hpp`, sample CSV datasets and expected output files.

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


