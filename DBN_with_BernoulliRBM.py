# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 23:56:48 2023

@author: BatuhanYILMAZ
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import joblib

# Step 1: Read the data
df = pd.read_csv("dataset.csv", delimiter=';')
# dataset is a random dataset
# as explained in the file, a Bernoulli-BernoulliRBM is more appropriate for classifications.
# for regressions, plesae use the toolbox link for matlab shared in the file and use Gaussian-BernoulliRBM type
# for regression tasks.

# Step 2: Determining the dependent and independent variable(s).
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardize the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: set the parameter grid to find the best combination
rbm1_params = {
    'rbm1__n_components': [50, 100, 150],
    'rbm1__learning_rate': [0.01, 0.05, 0.1],
    'rbm1__n_iter': [10, 20, 30],
    'rbm1__batch_size': [32, 64, 128]
}

rbm2_params = {
    'rbm2__n_components': [50, 100, 150],
    'rbm2__learning_rate': [0.01, 0.05, 0.1],
    'rbm2__n_iter': [10, 20, 30],
    'rbm2__batch_size': [32, 64, 128]
}

rbm3_params = {
    'rbm3__n_components': [50, 100, 150],
    'rbm3__learning_rate': [0.01, 0.05, 0.1],
    'rbm3__n_iter': [10, 20, 30],
    'rbm3__batch_size': [32, 64, 128]
}

# Define the linear regression model
LogisticRegression = LogisticRegression()

# Step 4: Create the pipeline and perform searching with cross-validation
grid_search = GridSearchCV(estimator=Pipeline(steps=[('rbm1', BernoulliRBM(random_state=101)),
                                                    ('rbm2', BernoulliRBM(random_state=101)),
                                                    ('rbm3', BernoulliRBM(random_state=101)),
                                                    ('LogisticRegression', LogisticRegression)]),
                           param_grid=[rbm1_params, rbm2_params, rbm3_params],
                           cv=3, n_jobs = -1)

# Step 5: Fit the model
grid_search.fit(X_train, y_train)

# Step 6: Find out the model properties
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

# Store best parameters and best score to a variable
best_params  = grid_search.best_params_
best_score = grid_search.best_score_

# Access to the best trained model with best parameter combination
best_model = grid_search.best_estimator_

# Get the weight matrices of the RBM layers for the best Model
rbm1_weights = best_model.named_steps['rbm1'].components_
rbm2_weights = best_model.named_steps['rbm2'].components_
rbm3_weights = best_model.named_steps['rbm3'].components_

# Create a list of the weight matrices
weight_list = [rbm1_weights, rbm2_weights, rbm3_weights]

# Get the biases for the hidden and visible layers of the best model for each RBM 
hidden_biases = best_model.named_steps['rbm1'].intercept_hidden_
visible_biases = best_model.named_steps['rbm1'].intercept_visible_

hidden_biases = best_model.named_steps['rbm2'].intercept_hidden_
visible_biases = best_model.named_steps['rbm2'].intercept_visible_

hidden_biases = best_model.named_steps['rbm3'].intercept_hidden_
visible_biases = best_model.named_steps['rbm3'].intercept_visible_

# Steo 7: Evaluation of the best model
y_pred = best_model.predict(X_test)

# Calculate the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print("Mean Squared Error (MSE): ", mse)
print("Mean Absolute Error (MAE): ", mae)
print("Root Mean Squared Error (RMSE): ", rmse)
print("R-squared (R2) Score: ", r2)

# Save the model
joblib.dump(best_model, 'model_DBN.pkl')

# Load the model
model = joblib.load('model_DBN.pkl')