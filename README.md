🚗 Car Purchase Amount Prediction

This project focuses on building a machine learning regression model to predict the total dollar amount a customer is willing to pay for a car based on their demographic and financial attributes.
The model can assist car sales teams in pricing strategy, customer targeting, and revenue forecasting.

Our Dataset 📊 includes the Following Properties of Car Purchasing Customers 

Customer’s Name	
Customer’s email address
Customer’s country
Gender	Male or Female
Customer’s age ( in years)
Annual Salary	Yearly income (in USD)
Outstanding credit card debt (in USD)
Net Worth	Total net worth (in USD)

Our Target Variable is The Car Purchase Amount

Detailed Approach Breakdown

🛠️ Data Preprocessing

Remove non-predictive columns (Name, Email)
Encode categorical features (Country, Gender)
Handle missing values
Feature scaling (StandardScaler / MinMaxScaler)

📊 Exploratory Data Analysis (EDA)

Analyze correlations between financial attributes and purchase amount
Visualize distributions and outliers

🧠 Model Development
Regression models such as:
Linear Regression
Random Forest Regressor
Gradient Boosting / XGBoost
Train-test split for evaluation

📉 Model Evaluation & Testing

🔹 Sample Prediction

A normalized test sample consisting of Gender, Age, Annual Salary, Credit Card Debt, and Net Worth is provided to the trained model.
The model predicts the normalized car purchase amount.
The predicted value is then inverse transformed to obtain the actual dollar amount.

🔹 Loss Analysis

Training and validation loss are plotted across epochs.
This helps analyze:
Model convergence
Training stability
Overfitting or underfitting behavior

🔹 Visualization
Line plots of training loss vs validation loss are used to monitor learning performance during model training.

🔮 Prediction:
  🚗 Predict car purchase amount for new customers based on input features
