# f1-champion-predictor
This project uses historical Formula 1 race pit stop data from 2018 to 2024 to build a predictive machine learning model capable of identifying potential race winners based on key performance metrics. The primary objective is to analyze driver behavior and pit stop strategies to forecast the 2025 F1 champion.

Key Features:
- Data Cleaning & Preprocessing:
- Removed missing values
- Converted tire compound types into numerical values
- Focused only on records with completed laps

Feature Engineering:
Selected important features such as:
- Average Pit Stop Time
- Tire Usage Aggression
- Driver Aggression Score
- Lap Time Variation

Visualization:
- Correlation heatmap to explore feature relationships
- Pairplot to observe class separation between winners and others

Model Training & Evaluation:
Applied four machine learning models:
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

Metrics used:
- Accuracy
- F1 Score
- Confusion Matrix for each model

Model Comparison:
- Printed a formatted model comparison table using tabulate, highlighting performance metrics.

Champion Prediction (2025):
The best-performing model is used to predict the most likely winner from the test set, simulating the 2025 champion prediction.
