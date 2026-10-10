# Random Forest Classifier on Iris Dataset

A clean implementation and analysis of **Random Forest (Ensemble Learning)** for multi-class classification on the classic Fisher's Iris dataset using **Scikit-Learn**.

---

## 📌 Project Overview

**Random Forest** is an ensemble learning method that constructs a collection of uncorrelated Decision Trees during training. By combining the predictions of multiple individual trees through bootstrap aggregation (**Bagging**) and random feature subsampling, Random Forests achieve significantly lower variance and greater resilience against overfitting compared to a single decision tree.

This project covers:
1. **Multi-Class Problem Formulation**: Classifying 3 distinct species of Iris flowers based on morphological measurements.
2. **Hyperparameter Tuning**: Inspecting the impact of `n_estimators` (number of trees) and `max_depth` (tree depth limits).
3. **Feature Importance Analysis**: Extracting Gini-based importance scores across Sepal and Petal attributes.
4. **Model Performance**: Multi-class accuracy evaluation across train and test sets.

---

## 🔬 Ensemble Architecture & Theory

### 1. Bootstrap Aggregation (Bagging)
For an ensemble of $B$ trees:
* Each tree $T_b$ is trained on an independently drawn bootstrap sample (with replacement) of size $N$ from the training dataset.
* For classification, the final ensemble prediction is decided by majority voting across all trees:
$$\hat{C}_{\text{RF}}(x) = \text{mode}\left\{ \hat{C}_b(x) \right\}_{b=1}^B$$

### 2. Random Subspace Feature Selection
* At each split, rather than considering all $p$ features, a random subset of $m \approx \sqrt{p}$ features is evaluated to select the best split.
* This decorrelates the individual trees, preventing strong dominant features from causing all trees to make identical splits.

### 3. Split Criterion (Gini Impurity)
$$I_G(p) = 1 - \sum_{k=1}^{K} p_k^2$$
* Measures label homogeneity at a given tree node across all $K$ classes.

---

## 📊 Dataset

* **Source**: Scikit-Learn built-in Iris dataset (`sklearn.datasets.load_iris`).
* **Samples**: 150 instances (50 samples per class).
* **Target Classes**:
  * `0`: Iris Setosa
  * `1`: Iris Versicolour
  * `2`: Iris Virginica
* **Features (4 continuous measurements in cm)**:
  1. Sepal Length
  2. Sepal Width
  3. Petal Length
  4. Petal Width

---

## 🛠️ Tech Stack & Requirements

* **Python 3.x**
* **Scikit-Learn**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RishiRaj1112/Random-Forest-Classifier-Iris.git
   cd Random-Forest-Classifier-Iris
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook RandomForestClassifier.ipynb
   ```

---

## 📈 Key Insights

* **Petal Dimensions Dominate**: Petal Length and Petal Width exhibit the highest feature importance in distinguishing Iris species.
* **Low Overfitting Risk**: Limiting `max_depth` controls model complexity while `n_estimators=100` guarantees smooth class boundaries.

---

## 👤 Author
* **Rishi Raj Talari** - [GitHub Profile](https://github.com/RishiRaj1112)
