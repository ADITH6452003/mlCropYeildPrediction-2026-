# 1. Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# 2. Load Dataset

data = pd.read_csv("/home/adith6452/Documents/ml_project/data/crop_yield.csv")
print(data.head())

# 3. Dataset Shape

print("Dataset Shape:", data.shape)

# 4. Crop Distribution

plt.figure(figsize=(10,6))
data['Crop'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Crops in Dataset")
plt.xlabel("Crop Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 5. Season Distribution

plt.figure(figsize=(8,5))
sns.countplot(x='Season', data=data)
plt.title("Crop Season Distribution")
plt.xticks(rotation=45)
plt.show()

# 6. State-wise Production

plt.figure(figsize=(12,6))
top_states = data.groupby("State")["Production"].sum().sort_values(ascending=False).head(10)
top_states.plot(kind='bar')
plt.title("Top 10 States by Crop Production")
plt.xlabel("State")
plt.ylabel("Total Production")
plt.show()

# 7. Yield Distribution

plt.figure(figsize=(8,5))
sns.histplot(data['Yield'], bins=30, kde=True)
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()

# 8. Rainfall vs Yield

plt.figure(figsize=(8,5))
sns.scatterplot(x="Annual_Rainfall", y="Yield", data=data)
plt.title("Rainfall vs Crop Yield")
plt.show()

# 9. Area vs Production

plt.figure(figsize=(8,5))
sns.scatterplot(x="Area", y="Production", data=data)
plt.title("Area vs Production")
plt.show()

# 10. Correlation Heatmap

plt.figure(figsize=(10,6))
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()