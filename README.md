# Crop Disease Prediction Experiment
# This program tries to predict crop diseases using weather and crop data


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap

# Step 1: Load the data files
osr_data = pd.read_excel('mean-time-series-osr-national-data-september-2024-1.xlsx', header=1)
wheat_data = pd.read_excel('mean-time-series-wheat-national-data-september-2024-2.xlsx', header=1)
weather_data = pd.read_csv('uk_daily_weather_by_region.csv', parse_dates=['date'])  # change path if needed

# Step 2: Clean the data
# Rename unclear column names to something easier
osr_data.rename(columns={'Unnamed: 0':'Crop', 'Unnamed: 1':'Survey_year'}, inplace=True)
wheat_data.rename(columns={'Unnamed: 0':'Crop', 'Unnamed: 1':'Survey_year'}, inplace=True)

# Step 3: Prepare weather data
# Make a new column for the year from the date
weather_data['year'] = weather_data['date'].dt.year

# Group weather data by region and year and calculate average and totals
weather_annual = weather_data.groupby(['region', 'year']).agg({
    'temperature_2m_mean': 'mean',
    'precipitation_sum': 'sum',
    'sunshine_duration': 'sum',
}).reset_index()

# Step 4: Combine weather data with crop disease data
merged_df = pd.merge(wheat_data, weather_annual, left_on=['Region', 'Survey_year'], right_on=['region', 'year'], how='inner')
merged_df.drop(columns=['region', 'year'], inplace=True)

# Step 5: Remove rows with missing information
merged_df.dropna(inplace=True)

# Step 6: Turn categories (like regions) into numbers that the computer can understand
ohe = OneHotEncoder(sparse=False)
region_encoded = ohe.fit_transform(merged_df[['Region']])
region_df = pd.DataFrame(region_encoded, columns=ohe.get_feature_names_out(['Region']))
merged_df = pd.concat([merged_df.reset_index(drop=True), region_df], axis=1)
merged_df.drop(columns=['Region'], inplace=True)

# Step 7: Scale the weather numbers so they are all between 0 and 1
num_cols = ['temperature_2m_mean', 'precipitation_sum', 'sunshine_duration']
scaler = MinMaxScaler()
merged_df[num_cols] = scaler.fit_transform(merged_df[num_cols])

# Step 8: Pick the columns for features and target (what we want to predict)
target_col = 'Zymoseptoria_tritici'  # the disease to predict
X = merged_df.drop(columns=[target_col])  # features (inputs)
y = merged_df[target_col]                 # target (output)

# Step 9: Split the data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 10: Define our machine learning models and parameters to try
rf = RandomForestRegressor(random_state=42)
rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}

xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]}

svm = SVR()
svm_params = {'kernel': ['rbf', 'linear'], 'C': [1, 10], 'epsilon': [0.1, 0.2]}

# Step 11: Function to train models and check how good they are
def train_and_evaluate(model, param_grid, model_name):
    print(f'Training {model_name}...')
    grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    print(f'Best settings for {model_name}: {grid.best_params_}')
    preds = grid.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    print(f'{model_name} RMSE (lower is better): {rmse:.4f}')
    print(f'{model_name} MAE (lower is better): {mae:.4f}')
    return grid.best_estimator_, preds

# Step 12: Train all models and get predictions
rf_model, rf_preds = train_and_evaluate(rf, rf_params, 'Random Forest')
xgb_model, xgb_preds = train_and_evaluate(xgb, xgb_params, 'XGBoost')
svm_model, svm_preds = train_and_evaluate(svm, svm_params, 'SVM')

# Step 13: Explain what features mattered the most using SHAP for XGBoost
print('Calculating feature importance with SHAP...')
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

# Show a summary plot of feature importance (this needs matplotlib)
shap.summary_plot(shap_values, X_test)

print("All done! Models trained and evaluated.")
