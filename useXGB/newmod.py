# 1. Import Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score

from xgboost import XGBRegressor

import matplotlib.pyplot as plt
from xgboost import plot_importance

# 2. Load Dataset

data = pd.read_csv("/home/adith6452/Documents/ml_project/data/crop_yield.csv")

print("Dataset Preview")
print(data.head())

print("\nDataset Info")
print(data.info())

# 3. Encode Categorical Data

le = LabelEncoder()

for col in data.select_dtypes(include=['object','string']).columns:
    data[col] = le.fit_transform(data[col])

# 4. Define Features and Target

X = data.drop("Yield", axis=1)
y = data["Yield"]

# 5. Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 6. Initialize XGBoost Model

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 7. Train Model

model.fit(X_train, y_train)

# 8. Predictions

y_pred = model.predict(X_test)

# 9. Model Evaluation

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calculate accuracy (percentage of predictions within acceptable range)
threshold = 0.1 * y_test.mean()  # 10% of mean yield
accurate_predictions = np.abs(y_test - y_pred) <= threshold
accuracy = (accurate_predictions.sum() / len(y_test)) * 100

print("\nModel Performance")
print("RMSE:", rmse)
print("R2 Score:", r2)
print(f"Accuracy: {accuracy:.2f}%")

# 10. Feature Importance

plt.figure(figsize=(10,6))
plot_importance(model)
plt.title("Feature Importance")
plt.show()

# 11. Example Prediction

sample = X_test.iloc[0:1]
prediction = model.predict(sample)
print("\nExample Prediction")
print("Predicted Yield:", prediction[0])
print("Actual Yield:", y_test.iloc[0])