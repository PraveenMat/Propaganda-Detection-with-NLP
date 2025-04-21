# Propaganda Detection with NLP

A project that applies two complementary Natural Language Processing approaches — **Bag-of-Words + SVM** and **LSTM neural networks** — to detect propaganda in news-style sentences and to classify specific propaganda techniques.

---

## 1. Overview

Propaganda poses a growing threat to information integrity. This repository demonstrates how traditional lexicographic models and sequence-based neural models can be trained and evaluated on the *SemEval-style* propaganda corpus. 

Automating propaganda detection helps journalists, fact-checkers, and platform engineers surface suspicious text and improve the reliability of information streams.

Two tasks are addressed:

| Task    | Objective                             | Labels                             |
|---------|---------------------------------------|------------------------------------|
| **Task 1** | Binary sentence-level classification | `propaganda` vs `not_propaganda`    |
| **Task 2** | Multi-class span-level technique identification | 8 propaganda techniques + `not_propaganda` |

### Dataset

- **Files:** `train.tsv`, `test.tsv` — tab-separated with two columns: `label` and `sentence` (propaganda spans delimited by `<BOS>` … `<EOS>`).
- **Total samples:** 2,414 sentences for training and 580 for testing.
- **Label taxonomy (9):**
  - `not_propaganda`
  - `flag_waving`
  - `appeal_to_fear_prejudice`
  - `causal_oversimplification`
  - `doubt`
  - `exaggeration_minimisation`
  - `loaded_language`
  - `name_calling_labelling`
  - `repetition`

### Class Distribution

| Propaganda Technique         | Train   | Test  |
|-----------------------------|---------|-------|
| `not_propaganda`            | **1,191** | **301** |
| `flag_waving`               | 148     | 39    |
| `appeal_to_fear_prejudice` | 151     | 43    |
| `causal_oversimplification`| 158     | 31    |
| `doubt`                     | 144     | 38    |
| `exaggeration_minimisation`| 164     | 28    |
| `loaded_language`          | 154     | 37    |
| `name_calling_labelling`   | 157     | 31    |
| `repetition`               | 147     | 32    |

> **Note:** The dataset exhibits a pronounced class imbalance, with `not_propaganda` making up ~49% of the training set while individual propaganda techniques average under 7% each. This imbalance motivates the choice of evaluation metrics (macro-F1) and informs future augmentation plans.

---

## 2. Methods

### Bag-of-Words + Linear SVC (scikit-learn)
- **Vectorizer:** `CountVectorizer` (min_df=2, unigrams).
- **Classifier:** Linear Support Vector Classifier with hinge loss.
- **Motivation:** Strong baseline for high-dimensional sparse text features; fast to train and inherently interpretable via feature weights.

### LSTM Neural Network (TensorFlow / Keras)
- **Tokenizer:** 10,000-word vocabulary, OOV token; sequences padded/truncated to 50 (Task 1) or 100 tokens (Task 2).
- **Architecture:** `Embedding → LSTM(64/128) → Dropout(0.4) → Dense` (sigmoid or softmax output).
- **Optimizer/Loss:** Adam (lr = 1e-3); binary cross-entropy (Task 1) or categorical cross-entropy (Task 2).
- **Training:** 70/30 train-validation split; batch = 20 (Task 1) or 10 (Task 2); 3–10 epochs.

---

## 3. Results

| Task | Model         | Accuracy | Notes                                                                 |
|------|---------------|:--------:|-----------------------------------------------------------------------|
| 1    | LSTM          | **0.73** | Higher recall & F1 for `not_propaganda`; modest overfitting mitigated by dropout. |
| 1    | BoW + SVC     | 0.65     | Slightly more balanced but larger FP/FN counts.                       |
| 2    | BoW + SVC     | **0.58** | Correctly recognises several techniques; best overall.                |
| 2    | LSTM          | 0.52     | Learns only `not_propaganda` class under current hyperparameters.    |

---

## 4. Key Takeaways

- **Simplicity can win:** A sparse Bag-of-Words representation paired with a linear SVM outperforms the LSTM on technique classification — proof that well-tuned classical models still deliver solid value on modest datasets.

- **Class imbalance matters:** The heavy skew toward `not_propaganda` (~49% of the training set) hinders the LSTM on the multi-class task and drives our choice of macro-F1 as the primary metric.

- **Interpretability trade-off:** The BoW-SVM exposes informative word weights that analysts can inspect, whereas the LSTM’s decisions are opaque without extra explainability tooling (e.g. SHAP, LIME).

---

## 5. Future Work

- **Bi-LSTM & Transformer baselines** (e.g. BERT) to capture bidirectional context.
- **Hybrid CNN-BiLSTM architecture** to exploit local n-gram features.
- **Data augmentation** (synonym replacement, back-translation) to address class imbalance.
- **Explainability tooling** (e.g. SHAP, LIME) for phrase-level insight.

---

## License

All Rights Reserved.
