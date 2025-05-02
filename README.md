# E-Commerce Product Performance Analysis

This project uses a synthetic yet realistic **E-Commerce Product Performance Dataset** to develop a regression model that predicts whether a product is likely to perform well in an online retail environment.

## 📊 Dataset Overview

The dataset contains **2,000 records** with the following features:

| Column Name          | Description |
|----------------------|-------------|
| `Product_Price`      | The listed price of the product in USD (range: $5 - $1000) |
| `Discount_Rate`      | Discount rate applied (0.0 to 0.8) |
| `Product_Rating`     | Customer rating (1 to 5) |
| `Number_of_Reviews`  | Total number of user reviews (0 to 5000, highly skewed) |
| `Stock_Availability` | 1 if available in stock, 0 otherwise |
| `Days_to_Deliver`    | Number of days it takes to deliver (1 to 30) |
| `Return_Rate`        | Proportion of items returned (0.0 to 0.9) |
| `Category_ID`        | ID of the product category (1 to 10) |

Note: Each column has approximately **5% missing values**, reflecting real-world data inconsistencies.

## 🎯 Project Goal

The objective is to predict a **Product Performance Score** using regression. The score will represent how well a product is expected to perform, which could be derived or defined using a combination of available metrics such as:
- Number of reviews
- Rating
- Return rate
- Price sensitivity

> You are free to define this performance metric using domain knowledge or a combination of target proxy features.

## 🔍 Tasks & Workflow

1. **Data Preprocessing**
   - Handle missing values
   - Normalize/transform skewed distributions
   - Feature engineering

2. **Exploratory Data Analysis (EDA)**
   - Correlation analysis
   - Distribution plots
   - Outlier detection

3. **Model Development**
   - Train/test split
   - Model selection (Linear Regression, Random Forest, etc.)
   - Evaluation metrics (R², RMSE, MAE)

4. **Model Interpretation**
   - Feature importance
   - SHAP or Partial Dependence Plots (optional)

## 🛠️ Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter or VS Code (recommended)

## 📁 Project Structure

```plaintext
.
├── data/
│   └── ecommerce_product_performance.csv
├── notebooks/
│   └── eda_and_modeling.ipynb
├── models/
│   └── saved_model.pkl
├── README.md
└── requirements.txt
