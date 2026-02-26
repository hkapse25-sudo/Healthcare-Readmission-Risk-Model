 Hospital Readmission Risk Prediction Model
Project Overview
This project focuses on identifying key risk factors for 30-day hospital readmissions by integrating Machine Learning and Classical Statistics. The goal is to provide healthcare providers with actionable insights to improve patient discharge planning and reduce early readmissions.
Technical Workflow
The project utilizes a dual-tool approach to ensure both predictive power and statistical validity:
•	Python (VS Code): Used to build a Random Forest Classifier to predict readmission risk based on patient demographics and comorbidities.
•	IBM SPSS: Used to perform Binary Logistic Regression to validate the statistical significance of the identified risk factors.
Key Findings
1. Model Performance
•	Overall Accuracy: The Random Forest model achieved a predictive accuracy of 89.0%.
•	Statistical Validation: The SPSS Logistic Regression confirmed a strong model fit with a Nagelkerke R Square of .829
2. Identified Risk Factors
•	Based on feature importance and odds ratios, the following factors were the most significant:
Factor	Python Importance	SPSS Exp(B) (Odds Ratio)	Statistical Sig. (p-value)
Comorbidity Index	0.491	20.773	< .001
Patient Age	0.305	1.168	< .001
Lab Procedures	0.105	1.000	0.990

•	Comorbidity Impact: Patients with higher comorbidity indices are significantly more likely to be readmitted (approx. 20 times more likely per unit increase).
•	Age Correlation: Risk of readmission increases steadily as the patient age increases.
•	![Readmission Risk by Age](Analysis_Results/Readmission_by_Age_Chart.png)
Tools & Libraries
•	Languages: Python 3.13.
•	Libraries: pandas, NumPy, scikit-learn, matplotlib.
•	Software: VS Code, IBM SPSS Statistics.
 

🛠 Methodology
The analysis followed a multi-stage workflow to ensure the reliability of the readmission risk predictions.
Phase 1: Data Generation & Preprocessing (Python)
•	Synthetic Data Creation: A clinical dataset of 1,000 patient records was generated using NumPy and Pandas.
•	Feature Engineering: Variables included patient age, gender, comorbidity index, number of lab procedures, and number of medications.
•	Target Logic: A 30-day readmission flag was created using a weighted risk probability based on clinical correlations between age and chronic conditions.
Phase 2: Predictive Modeling (VS Code)
•	Algorithm: A Random Forest Classifier was implemented using scikit-learn.
•	Training: The data was split into training (80%) and testing (20%) sets to evaluate performance.
•	Feature Importance: The model ranked the comorbidity index (49.1%) and age (30.5%) as the most influential factors for predicting readmission.
Phase 3: Statistical Validation (IBM SPSS)
•	Data Integration: The generated readmission_analysis.csv was imported into SPSS for rigorous statistical testing.
•	Binary Logistic Regression: This was performed to validate the machine learning findings and calculate the Odds Ratio (Exp(B)).
•	Significance Testing: Both primary risk factors achieved a p-value of <.001, confirming they are statistically significant predictors.
Phase 4: Visualization
•	Python: Generated a feature importance horizontal bar chart to visualize model logic.
•	SPSS: Created a Clustered Bar Chart to display the percentage of readmissions across different age categories, providing a clear demographic view of risk.

Summary :
By including the SPSS files, you are confirming the impressive stats you've already found:
•	Accuracy: 90.0%.
•	Top Predictor: Comorbidity Index with an Exp(B) of 20.773.
•	Significance: All primary factors show a P-value of <.001.


