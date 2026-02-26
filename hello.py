import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

print("--- Starting Project: Readmission Risk Prediction ---")

# Define the absolute path to your Desktop folder
# This ensures Python and SPSS always look in the same place
file_path = r'C:\Users\harsh\Desktop\Python\readmission_analysis.csv'

# 1. Create a Realistic Healthcare Dataset
print("Step 1: Generating clinical dataset...")
np.random.seed(42)
n_patients = 1000

data = {
    'patient_id': range(1, n_patients + 1),
    'age': np.random.randint(18, 95, n_patients),
    'gender': np.random.choice(['Male', 'Female'], n_patients),
    'comorbidity_index': np.random.randint(0, 5, n_patients),
    'primary_diagnosis': np.random.choice(['Circulatory', 'Respiratory', 'Diabetes', 'Digestive', 'Other'], n_patients),
    'num_lab_procedures': np.random.randint(1, 100, n_patients),
    'num_medications': np.random.randint(1, 30, n_patients)
}

df = pd.DataFrame(data)

# Logic: Risk increases with Age and Comorbidities
risk_prob = (df['age'] / 200) + (df['comorbidity_index'] / 10) + (np.random.rand(n_patients) * 0.2)
df['readmission_30_days'] = (risk_prob > 0.6).astype(int)

# 2. Save the file to your Desktop
df.to_csv(file_path, index=False)
print(f"--- Success! File saved to: {file_path} ---")

# 3. Load the data back for Machine Learning
print("\nStep 2: Loading data for model training...")
df_ml = pd.read_csv(file_path)

# Prepare Features (X) and Target (y)
X = df_ml[['age', 'comorbidity_index', 'num_lab_procedures', 'num_medications']]
y = df_ml['readmission_30_days']

# 4. Split and Train the Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Output the Results
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Factor': X.columns, 
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Model Results: Identifying Key Risk Factors ---")
print(feature_importance_df)

accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.2%}")
print("\nYou can now open this file in SPSS to validate these factors.")
import matplotlib.pyplot as plt

# Sort feature importances in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# Create the plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Factor'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.title('Key Risk Factors for Hospital Readmission')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Save the plot as an image for your report
plt.savefig(r'C:\Users\harsh\Desktop\Python\risk_factors_chart.png')
plt.show()
