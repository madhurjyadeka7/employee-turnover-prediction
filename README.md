# Employee Turnover Prediction

A Machine Learning project that predicts whether an employee may leave an organization using Logistic Regression and a Streamlit web application.

## Project Overview

This project compares three Logistic Regression models:
- Baseline Logistic Regression
- L1-Regularized Logistic Regression (Lasso)
- L2-Regularized Logistic Regression (Ridge)

The models are evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

## Features

- Interactive Streamlit dashboard
- Employee feature inputs
- Turnover prediction and probability display
- Comparison of Logistic Regression models

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Model Results

The L1-Regularized Logistic Regression model achieved the following results on the project's evaluation split:

| Metric | Score |
|---|---:|
| Accuracy | 87.04% |
| Precision | 88.14% |
| Recall | 83.20% |
| F1-score | 85.60% |

These results are from the project's existing evaluation and may vary with different data splits.

## Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/madhurjyadeka7/employee-turnover-prediction.git
   cd employee-turnover-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Disclaimer

This is an educational machine learning project. Predictions should not be used as the sole basis for real-world hiring, employee evaluation, or termination decisions.

## Author

Madhurjya Deka

GitHub: [@madhurjyadeka7](https://github.com/madhurjyadeka7)
