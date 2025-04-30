# Logistic Regression on Diabetes Dataset with SMOTE

This project demonstrates a basic machine learning pipeline using Logistic Regression on a diabetes dataset. It includes data preprocessing, handling class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**, training a logistic regression model, and evaluating its performance.

## 📁 Dataset

The dataset used is `diabetes.csv`, which includes diagnostic measurements such as glucose level, BMI, and age, along with a target variable indicating the presence of diabetes.

### Features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target variable: 0 = No diabetes, 1 = Diabetes)

## ⚙️ Workflow

1. **Data Loading & Exploration**  
   Load the CSV data using pandas and inspect for missing or anomalous values.

2. **Preprocessing**  
   - Feature normalization or standardization (optional)
   - Train-test split

3. **Handling Class Imbalance with SMOTE**  
   - Before SMOTE: Check class distribution of `Outcome`
   - Apply SMOTE to oversample the minority class
   - After SMOTE: Validate balanced class distribution

4. **Logistic Regression**  
   - Train logistic regression on balanced dataset
   - Evaluate using accuracy, precision, recall, F1 score, and confusion matrix

5. **Visualization (optional)**  
   - Confusion matrix heatmap
   - ROC curve

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/diabetes-logistic-smote.git
cd diabetes-logistic-smote
pip install -r requirements.txt
```

## 📦 Dependencies

- pandas  
- numpy  
- scikit-learn  
- imbalanced-learn  
- matplotlib  
- seaborn

Install using:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

## 🚀 Usage

```python
python main.py
```

## 📊 Results

Without SMOTE:
- Class distribution skewed
- Lower recall on minority class (diabetes)

With SMOTE:
- Improved balance and recall
- More robust performance across metrics

## 📁 File Structure

```
.
├── diabetes.csv
├── main.py
├── README.md
└── requirements.txt
```

