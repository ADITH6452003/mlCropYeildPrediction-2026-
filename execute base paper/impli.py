# Import libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Load dataset
df = pd.read_csv("/home/adith6452/Documents/ml_project/data/crop_yield.csv")

print(df.head())
print(df.info())


# DATA PREPROCESSING

# Handle missing values
df = df.dropna()


# Encode categorical columns
label_encoder = LabelEncoder()

categorical_cols = ['Crop', 'Season', 'State']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])


# CREATE TARGET VARIABLE

# Calculate yield if not present
if 'Yield' not in df.columns:
    df['Yield'] = df['Production'] / df['Area']


# Convert yield into 4 classes (quartiles)
# FIX: convert to integer
df['Yield_Class'] = pd.qcut(df['Yield'], 4, labels=[0,1,2,3]).astype(int)


# FEATURE SELECTION

features = [
    'Crop', 'Crop_Year', 'Season',
    'State', 'Area', 'Production',
    'Annual_Rainfall', 'Fertilizer', 'Pesticide'
]

X = df[features]
y = df['Yield_Class']


# NORMALIZATION

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)


# MODELS USED IN PAPER

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(),

    "Random Forest": RandomForestClassifier(n_estimators=100),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes": GaussianNB(),

    "SVM": SVC(),

    "Gradient Boosting": GradientBoostingClassifier()
}


# TRAINING + EVALUATION

for name, model in models.items():

    print("\n======================")
    print(name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))