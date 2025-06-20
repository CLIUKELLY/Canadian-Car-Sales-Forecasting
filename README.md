# Canadian Car Sales Forecasting

Welcome to the **Canadian Car Sales Forecasting** repository! This project focuses on predicting motor vehicle sales in Canada using various time series forecasting models. The dataset is sourced from Statistics Canada (StatsCan), and the repository implements multiple forecasting techniques to compare their performance.

## Project Overview

This repository contains code to analyze and forecast motor vehicle sales data using the following models:
- **ARIMA**: A traditional statistical model for time series forecasting.
- **NeuralProphet**: A neural network-based forecasting framework with Prophet-like capabilities.
- **Prophet**: A forecasting library developed by Facebook, designed for time series with seasonality.
- **LSTM (Long Short-Term Memory)**: A deep learning model for capturing complex temporal patterns.

The goal is to evaluate and compare the performance of these models using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

## Dataset

- **Source**: Statistics Canada (StatsCan)
- **File**: `New motor vehicle sales data.csv`
- **Description**: The dataset includes monthly sales data for new motor vehicles in Canada. The data is filtered to focus on national-level, unadjusted sales for all vehicle types and origins of manufacture.
- **Time Range**: The analysis focuses on the last 144 months of data, with the final 36 months used as the test set.

## Repository Structure

- **`New motor vehicle sales data.csv`**: The raw dataset from StatsCan.
- **`forecasting_models.py`**: The main script containing data preprocessing, model training, forecasting, and evaluation.
- **`README.md`**: This file, providing an overview of the project.