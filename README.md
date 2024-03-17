# Flight Price Prediction

## Overview

This project focuses on predicting flight prices using various machine learning models. The dataset contains features such as airline, source, destination, duration, total stops, and additional information. The project involves exploratory data analysis (EDA), feature engineering, and model training to accurately predict flight prices.

## Dataset

The dataset used in this project contains information about flight details and prices. [Data Source](https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh/code?datasetId=140442&sortBy=dateCreated)

Key features include:

- Airline: The airline operating the flight
- Source: The starting point of the flight
- Destination: The endpoint of the flight
- Duration: Total duration of the flight
- Total_Stops: Number of stops between the source and destination
- Additional_Info: Additional information about the flight
- Price: Price of the flight ticket
  
## Libraries Used

- pandas
- matplotlib
- seaborn
- numpy
- scikit-learn
- xgboost

## Key Steps

1. **Exploratory Data Analysis:** Analyze the data to understand the distribution of various features and their relationship with flight prices.
2. **Feature Engineering:** Create new features such as departure and arrival times, flight duration in minutes, and frequency encoding for the route.
3. **Modeling and Prediction:** Train various regression models to predict flight prices, evaluate their performance and use the best-performing model to make predictions on the test dataset.

## Models Used

- Linear Regression
- ElasticNet
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- Support Vector Regressor (SVR)
- XGBoost Regressor

## Results

The models are evaluated based on their performance metrics such as RMSE (Root Mean Squared Error) and R-squared score. The XGBoost Regressor is found to be the best-performing model for this dataset.

| Model                         | RMSE     | R2 Score  | Score    |
|-------------------------------|----------|-----------|----------|
| Lasso                         | 0.364501 | -0.965990 | 0.404442 |
| ElasticNet                    | 0.361464 | -0.510280 | 0.414326 |
| RidgeCV                       | 0.305209 |  0.271923 | 0.582439 |
| LinearRegression              | 0.305209 |  0.271934 | 0.582439 |
| SVR                           | 0.294032 |  0.488338 | 0.612462 |
| AdaBoostRegressor             | 0.255574 |  0.274876 | 0.707208 |
| GradientBoostingRegressor     | 0.164851 |  0.851156 | 0.878182 |
| DecisionTreeRegressor         | 0.146908 |  0.903038 | 0.903258 |
| RandomForestRegressor         | 0.113365 |  0.939124 | 0.942392 |
| XGBRegressor                  | 0.106426 |  0.946901 | 0.949228 |

## Conclusion

The project demonstrates the use of various machine learning models to predict flight prices. Feature engineering plays a crucial role in improving model performance. The XGBoost Regressor is identified as the most effective model for predicting flight prices in this case.

