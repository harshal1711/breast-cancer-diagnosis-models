## Project: Breast Cancer Classification Using ML

This project builds and evaluates three different classification models—**Decision Tree**, **Logistic Regression**, and **K-Nearest Neighbors (KNN)**—to predict whether a tumor is malignant or benign using the UCI Breast Cancer Wisconsin dataset.

---

### Key Highlights

- **Dataset**: [UCI Breast Cancer Wisconsin Diagnostic Data](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **Target Variable**: Diagnosis (`M` = Malignant, `B` = Benign; encoded as 1 and 0)
- **Features Used**: 30 real-valued features representing mean, standard error, and "worst" values of attributes

---

### Preprocessing & Feature Engineering

- **Correlated Feature Removal**: Performed pairwise correlation checks using a threshold of 0.80. Dropped highly correlated pairs (e.g., `radius` vs `perimeter`, `area`, `worst_area`, etc.) to reduce multicollinearity.
- **Outlier Detection**: Applied the IQR method to flag and examine features with excessive outliers, such as `SE_radius` and `SE_smoothness`.
- **Scaling**: MinMaxScaler was applied across all features to standardize inputs before training models.
- **Stratified Train/Val/Test Split**: Ensured balanced splits across the target class for fair evaluation.

---

### Models Evaluated

- **Decision Tree**
  - Tuned `max_depth`, `min_samples_split`, and `min_samples_leaf`
  - Evaluated performance across depth values from 1 to 19
  - Achieved strong recall but began overfitting at higher depths

- **K-Nearest Neighbors (KNN)**
  - Explored odd `k` values from 1 to 29 using Euclidean distance
  - Best performance at `k=15`
  - Moderate recall on malignant cases

- **Logistic Regression**
  - Compared L1 (Lasso) vs L2 (Ridge) penalties
  - Tuned `C` across [100, 10, 1, 0.1]
  - L1 with `C=100` gave the best test recall and F1-score

---

### Model Performance (Test Set)

| Model                        | Accuracy | Malignant Recall | F1-Score |
|-----------------------------|----------|------------------|----------|
| Decision Tree (depth=3, min_leaf=20) | 95%      | 93%              | 0.93     |
| KNN (k=15)                   | 91%      | 88%              | 0.88     |
| Logistic Regression (L1, C=100) | **96%** | **98%**         | **0.95** |

---

###  Key Takeaways

- Logistic Regression with L1 regularization performed the best, especially in maximizing recall for malignant cases (critical in cancer prediction).
- Correlation-based feature pruning improved generalization and helped reduce noise in KNN and tree-based models.
- Decision Trees offer interpretability but risk overfitting with deeper depths.
- KNN was sensitive to `k` but achieved decent results with properly scaled features.

---

