# Complete Machine Learning Workflow

## 1. Data Loading and Initial Inspection
- **Load the dataset** using libraries like Pandas:
  - Example: `import pandas as pd; df = pd.read_csv('data.csv')`
- **Inspect the data**:
  - Check shape: `df.shape` (rows and columns).
  - View first few rows: `df.head()`.
  - Data types: `df.dtypes`.
  - Summary statistics: `df.describe()` (for numerical features).
  - Unique values in categorical columns: `df['column'].unique()`.
- **Why this step?**
  - Understand the structure, identify potential issues like data types mismatches or outliers early.

## 2. Exploratory Data Analysis (EDA)
- **Visualize distributions**:
  - Histograms for numerical features: `df['feature'].hist()`.
  - Box plots for outliers: `df.boxplot(column='feature')`.
  - Count plots for categorical features: `import seaborn as sns; sns.countplot(x='category', data=df)`.
- **Analyze relationships**:
  - Correlation matrix: `df.corr()` and heatmap: `sns.heatmap(df.corr(), annot=True)`.
  - Scatter plots: `sns.scatterplot(x='feature1', y='feature2', data=df)`.
  - Pair plots: `sns.pairplot(df)`.
- **Check for class imbalance** (in classification): `df['target'].value_counts()`.
- **Why EDA?**
  - Uncover patterns, anomalies, correlations, and insights that guide preprocessing and feature engineering.
  - Helps in hypothesis generation and avoiding surprises later.

## 3. Data Preprocessing and Cleaning
- **Handle missing values**:
  - Drop rows/columns: `df.dropna()` or `df.dropna(axis=1)`.
  - Impute: Mean/median for numerical: `df['feature'].fillna(df['feature'].mean())`.
  - Mode for categorical: `df['category'].fillna(df['category'].mode()[0])`.
- **Encode categorical variables**:
  - One-hot encoding: `pd.get_dummies(df, columns=['category'])`.
  - Label encoding: `from sklearn.preprocessing import LabelEncoder; le = LabelEncoder(); df['category'] = le.fit_transform(df['category'])`.
- **Handle outliers**:
  - Detect using IQR: `Q1 = df['feature'].quantile(0.25); Q3 = df['feature'].quantile(0.75); IQR = Q3 - Q1`.
  - Remove: `df = df[~((df['feature'] < (Q1 - 1.5 * IQR)) | (df['feature'] > (Q3 + 1.5 * IQR)))]`.
- **Feature scaling/normalization**:
  - StandardScaler: `from sklearn.preprocessing import StandardScaler; scaler = StandardScaler(); df_scaled = scaler.fit_transform(df)`.
  - MinMaxScaler: `from sklearn.preprocessing import MinMaxScaler; scaler = MinMaxScaler(); df_normalized = scaler.fit_transform(df)`.
- **Why preprocessing?**
  - Ensures data is clean, consistent, and suitable for modeling (e.g., algorithms like SVM or KNN are sensitive to scale).

## 4. Feature Engineering and Selection
- **Create new features**:
  - Derived features: e.g., `df['bmi'] = df['weight'] / (df['height'] ** 2)`.
  - Binning: `pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young', 'Adult', 'Senior'])`.
  - Interactions: `df['interaction'] = df['feature1'] * df['feature2']`.
- **Feature selection methods**:
  - Filter methods: Correlation-based: Select top k: `from sklearn.feature_selection import SelectKBest, f_classif; selector = SelectKBest(f_classif, k=10); X_selected = selector.fit_transform(X, y)`.
  - Wrapper methods: Recursive Feature Elimination (RFE): `from sklearn.feature_selection import RFE; from sklearn.linear_model import LogisticRegression; rfe = RFE(LogisticRegression(), n_features_to_select=5); X_rfe = rfe.fit_transform(X, y)`.
  - Embedded methods: Lasso regularization in models.
- **Dimensionality reduction**:
  - PCA: `from sklearn.decomposition import PCA; pca = PCA(n_components=2); X_pca = pca.fit_transform(X)`.
  - t-SNE or UMAP for visualization.
- **Why feature engineering/selection?**
  - Improves model performance by reducing noise, handling multicollinearity, and focusing on relevant features.
  - Reduces overfitting and computational cost.

## 5. Data Splitting
- **Split the dataset** into:
  - `X`: All features (after preprocessing and selection).
  - `y`: Labels/targets.
  - Then: `from sklearn.model_selection import train_test_split; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`.
- **Optional: Validation set** for hyperparameter tuning: `X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)`.
- **Stratified split** for imbalanced data: `train_test_split(X, y, test_size=0.2, stratify=y)`.
- **Why this split?**
  - Training data (`X_train`, `y_train`) teaches the model.
  - Validation data tunes hyperparameters.
  - Test data (`X_test`, `y_test`) evaluates generalization on unseen data.

## 6. Model Initialization and Hyperparameter Tuning
- **Choose an algorithm** (e.g., `from sklearn.linear_model import LogisticRegression; model = LogisticRegression()` or `from sklearn.ensemble import RandomForestClassifier; model = RandomForestClassifier()`).
- **Define hyperparameters**:
  - Examples: `max_depth=10`, `n_estimators=100`, `C=1.0` (regularization).
  - Hyperparameters control complexity and are set before training.
- **Hyperparameter tuning**:
  - Grid Search: `from sklearn.model_selection import GridSearchCV; param_grid = {'C': [0.1, 1, 10]}; grid = GridSearchCV(model, param_grid, cv=5); grid.fit(X_train, y_train)`.
  - Random Search: `from sklearn.model_selection import RandomizedSearchCV`.
  - Use cross-validation: `cv=5` for k-fold.
- **Why tuning?**
  - Optimizes model performance; default hyperparameters may not be ideal.

## 7. Model Training (`fit`)
- **Syntax:** `model.fit(X_train, y_train)` (or `grid.best_estimator_.fit(X_train, y_train)` after tuning).
- **What happens internally?**
  - Model processes feature-label pairs.
  - Learns patterns (e.g., optimizes weights via gradient descent, builds decision trees).
  - Parameters like coefficients or thresholds are updated.
- **Handle class imbalance** during training: Use `class_weight='balanced'` or oversampling (SMOTE: `from imblearn.over_sampling import SMOTE; smote = SMOTE(); X_train_res, y_train_res = smote.fit_resample(X_train, y_train)`).
- **After this step**, the model is "trained" and ready for inference.

## 8. Predictions (`predict`)
- **Syntax:** `y_pred = model.predict(X_test)` or `y_pred_val = model.predict(X_val)` for validation.
- **Process:**
  - Input test/validation features into the trained model.
  - Applies learned parameters to generate outputs.
- **Probabilities** (if needed): `y_prob = model.predict_proba(X_test)` for classification confidence.
- **Predictions** are class labels (classification) or continuous values (regression).

## 9. Model Evaluation
- **Compare predictions** with true labels: `y_pred` vs. `y_test`.
- **Metrics for classification**:
  - Accuracy: `from sklearn.metrics import accuracy_score; accuracy_score(y_test, y_pred)`.
  - Precision, Recall, F1: `precision_score(y_test, y_pred)`, etc.
  - Confusion Matrix: `confusion_matrix(y_test, y_pred)`.
  - ROC-AUC: `roc_auc_score(y_test, y_prob)`.
- **Metrics for regression**:
  - MAE: `mean_absolute_error(y_test, y_pred)`.
  - MSE/RMSE: `mean_squared_error(y_test, y_pred)`, `sqrt(mse)`.
  - R²: `r2_score(y_test, y_pred)`.
- **Cross-validation scores**: `from sklearn.model_selection import cross_val_score; cross_val_score(model, X, y, cv=5)`.
- **Why evaluation?**
  - Quantifies performance, identifies weaknesses (e.g., bias/variance), and compares models.

## 10. Model Interpretation and Deployment
- **Interpret the model**:
  - Feature importance: `model.feature_importances_` (for tree-based models).
  - SHAP/LIME: `import shap; explainer = shap.Explainer(model); shap_values = explainer(X_test)`.
- **Iterate if needed**: Go back to EDA/preprocessing based on insights.
- **Deployment**:
  - Save model: `import joblib; joblib.dump(model, 'model.pkl')`.
  - Load and predict: `loaded_model = joblib.load('model.pkl'); loaded_model.predict(new_data)`.
  - Use frameworks like Flask/Docker for API deployment.
- **Why this step?**
  - Ensures model is explainable, reliable, and usable in production.
 
**Prepared by [Ben Gregory John](https://github.com/BenGJ10)**
