# fraud-detection-ecommerce-banking


## Overview
Adey Innovations Inc., a leading financial technology company, aims to enhance fraud detection in e-commerce and banking transactions. By leveraging advanced machine learning models and geolocation analysis, this project improves fraud detection accuracy, reduces financial losses, and strengthens transaction security. The system enables real-time monitoring and reporting, ensuring businesses can respond swiftly to fraudulent activities.

## Features
- **Data Analysis & Preprocessing**: Handling missing values, merging datasets, feature engineering.
- **Machine Learning Models**: Training and evaluating models like Logistic Regression, Random Forest, Gradient Boosting, MLP, CNN, RNN, and LSTM.
- **Model Explainability**: Using SHAP and LIME for understanding model predictions.
- **API Development**: Creating a Flask-based REST API to serve fraud detection models.
- **Dockerization**: Deploying the API in a Docker container for scalability.
- **Interactive Dashboard**: Using Flask and Dash to visualize fraud insights.

## Datasets Used
1. **Fraud_Data.csv**: Contains e-commerce transaction data.
2. **IpAddress_to_Country.csv**: Maps IP addresses to their respective countries.
3. **creditcard.csv**: Contains anonymized bank transaction data for fraud detection.

## Project Workflow
### **1. Data Analysis & Preprocessing**
- Handle missing values
- Data cleaning (removing duplicates, correcting data types)
- Exploratory Data Analysis (EDA)
- Merge datasets for geolocation analysis
- Feature engineering (transaction velocity, time-based features)
- Encode categorical features

### **2. Model Building & Training**
- Train-test split of datasets
- Model selection: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, MLP, CNN, RNN, LSTM
- Model evaluation using performance metrics

### **3. Model Explainability**
- **SHAP**: Understanding feature importance with summary, force, and dependence plots.
- **LIME**: Explaining individual predictions with feature importance plots.

### **4. Model Deployment & API Development**
- Develop a Flask API to serve the fraud detection models
- Create endpoints for prediction and logging requests
- Dockerize the application for scalable deployment

### **5. Dashboard Development**
- Develop an interactive dashboard using Dash
- Display fraud insights, transaction statistics, and geolocation-based fraud analysis

## Installation & Setup
### **1. Clone the Repository**
```sh
git clone https://github.com/Roman0290/fraud-detection-ecommerce-banking.git
cd fraud-detection-ecommerce-banking
