# ML Models from Scratch

This repository contains implementations of fundamental **Machine Learning (ML) algorithms from scratch** using Python. The primary objective is to deeply understand the underlying mathematics, logic, and step-by-step processes of these algorithms without depending on high-level ML libraries like `scikit-learn`.  

By building these models manually, we gain better insights into how they work, their assumptions, limitations, and how they can be extended or optimized in real-world applications.

---
## Implemented Models

### 1. Linear Regression
**Definition:**  
Linear Regression is a supervised learning algorithm used for modeling the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a straight line (in higher dimensions, a hyperplane).

**Key Points:**
- Uses the concept of minimizing the **Mean Squared Error (MSE)**.
- Parameters are optimized using **Gradient Descent** (or Normal Equation for closed-form solutions).
- Assumes a **linear relationship** between features and the target.

**Use Cases:**
- Predicting housing prices.
- Estimating sales based on advertising spend.
- Forecasting continuous outcomes like stock prices, temperatures, or demand.

---

### 2. Logistic Regression
**Definition:**  
Logistic Regression is a supervised learning algorithm used for **binary classification tasks**. Instead of predicting continuous values, it models the probability that an instance belongs to a particular class using the **sigmoid function**.

**Key Points:**
- Outputs probabilities between 0 and 1.
- Decision boundaries are created using a threshold (commonly 0.5).
- Optimized using **Maximum Likelihood Estimation (MLE)** and solved with **Gradient Descent**.
- Can be extended to multi-class problems using techniques like **One-vs-Rest (OvR)** or **Softmax Regression**.

**Use Cases:**
- Spam email detection.
- Customer churn prediction.
- Disease diagnosis (e.g., predicting diabetes presence).
- Credit risk assessment.

---

### 3. Principal Component Analysis (PCA)
**Definition:**  
PCA is an **unsupervised dimensionality reduction technique** used to project high-dimensional data into a lower-dimensional space while preserving as much variance as possible.

**Key Points:**
- Based on **eigenvalues and eigenvectors** of the covariance matrix.
- Helps remove multicollinearity by transforming features into uncorrelated principal components.
- Reduces computation cost and noise in the dataset.
- Not a predictive model, but often a **preprocessing step**.

**Use Cases:**
- Visualization of high-dimensional datasets (e.g., reducing 100 features to 2D/3D).
- Compression of large datasets.
- Noise reduction in signals or images.
- Speeding up machine learning algorithms by reducing input size.

---

## Why Build from Scratch?
- To strengthen mathematical foundations in ML (linear algebra, calculus, probability, statistics).
- To gain complete transparency in model workings (beyond the abstraction of libraries).
- To understand optimization challenges (e.g., convergence issues in gradient descent).
- To build intuition for extending models or debugging issues in real-world scenarios.

---

## References

- Pattern Recognition and Machine Learning by Christopher M. Bishop

- Geeks for Geeks, Analytics Vidhya, Towards Data Science, Kaggle , Medium 

