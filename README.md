# Insurance Charges Prediction

Predicting medical insurance charges using EDA and a Linear Regression model, 
built on the classic Kaggle medical insurance dataset.

## Overview
This project explores what factors drive medical insurance costs and builds 
a regression model to predict charges based on personal attributes like age, 
BMI, smoking status, and region.

## Key Finding
**Smoking status is by far the strongest driver of insurance charges** 
(correlation of 0.79 with charges) — far ahead of age (0.3) or BMI (0.2). 
Smokers pay dramatically more on average, confirmed both statistically and 
visually via boxplot analysis.

## What I Did
- Exploratory Data Analysis (distribution of charges, missing values, duplicates)
- Visualized relationships (boxplots, scatterplots, correlation heatmap)
- Encoded categorical features (label encoding + one-hot encoding)
- Engineered a new feature (`bmi_age`, interaction of BMI and age)
- Trained a Linear Regression model with an 80/20 train-test split
- Evaluated using Mean Absolute Error and Actual vs Predicted plots
- Analyzed feature importance via model coefficients

## Results
- **Mean Absolute Error:** ~$4,188
- Actual vs Predicted plot shows strong performance for non-smokers, 
  with more error/spread for high-cost smoker cases — a clear direction 
  for future improvement (e.g. non-linear models).

## Tools Used
- Python, pandas, numpy
- seaborn, matplotlib (visualization)
- scikit-learn (modeling)

## How to Run
1. Clone this repo
2. Install requirements: `pip install pandas numpy seaborn matplotlib scikit-learn`
3. Open `notebook.ipynb` in Jupyter or Colab and run all cells

## Dataset
Classic medical insurance cost dataset (1338 records, 7 features: age, sex, 
bmi, children, smoker, region, charges).
