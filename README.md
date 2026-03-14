# 🏥 Hospital Readmission Risk Prediction Model

A Machine Learning and Statistical Analysis project focused on predicting **30-day hospital readmission risk** using patient demographic and clinical factors.

This project combines **predictive machine learning** with **classical statistical validation** to identify the most influential drivers of hospital readmissions and provide insights that can help improve **patient discharge planning and healthcare outcomes**.

---

# 📌 Project Overview

Hospital readmissions are a major challenge for healthcare systems worldwide. Early identification of high-risk patients can help healthcare providers implement preventative strategies and reduce avoidable readmissions.

This project integrates:

- 🤖 **Machine Learning (Random Forest Classifier)**
- 📊 **Statistical Modeling (Binary Logistic Regression)**

The goal is to combine **predictive performance** with **statistical interpretability**.

---

# ⚙️ Technical Workflow

The project uses a **dual-tool analytical framework** to ensure both predictive accuracy and statistical reliability.

| Tool | Purpose |
|-----|------|
| 🐍 **Python (VS Code)** | Building the Random Forest model to predict readmission risk |
| 📈 **IBM SPSS** | Validating predictors using Binary Logistic Regression |

---

# 📊 Key Findings

## 1️⃣ Model Performance

| Metric | Result |
|------|------|
| 🎯 Random Forest Accuracy | **89.0%** |
| 📈 Logistic Regression Fit | **Nagelkerke R² = 0.829** |

These results demonstrate strong predictive power and statistical reliability.

---

## 2️⃣ Identified Risk Factors

The following features were identified as the **most influential predictors of hospital readmission**:

| Factor | Python Importance | SPSS Exp(B) (Odds Ratio) | P-Value |
|------|------|------|------|
| **Comorbidity Index** | 0.491 | 20.773 | < .001 |
| **Patient Age** | 0.305 | 1.168 | < .001 |
| **Lab Procedures** | 0.105 | 1.000 | 0.990 |

### 🔍 Interpretation

**Comorbidity Index**
- Strongest predictor of readmission
- Patients with higher comorbidity scores are **~20x more likely to be readmitted**

**Patient Age**
- Readmission risk increases gradually with **increasing age**

**Lab Procedures**
- Minimal predictive contribution in this dataset

---

# 📉 Visualization

### Readmission Risk by Age

![Clinical Readmission Analysis](readmission_age_chart.png)

This visualization highlights the **trend of increasing readmission risk across older age groups**.

---

# 🛠 Tools & Technologies

### Programming
- 🐍 **Python 3.13**

### Python Libraries
- `pandas`
- `NumPy`
- `scikit-learn`
- `matplotlib`

### Software
- 💻 **VS Code**
- 📊 **IBM SPSS Statistics**

---

# 🔬 Methodology

The analysis followed a **multi-stage workflow** to ensure reliability and reproducibility.

---

## Phase 1: Data Generation & Preprocessing (Python)

- 📦 **Synthetic Dataset Creation**
  - Generated **1,000 patient records** using NumPy and Pandas.

- 🧩 **Feature Engineering**
  - Variables included:
  - Patient Age
  - Gender
  - Comorbidity Index
  - Number of Lab Procedures
  - Number of Medications

- 🎯 **Target Variable**
  - A **30-day readmission flag** was created using weighted clinical risk probabilities based on age and comorbidities.

---

## Phase 2: Predictive Modeling (VS Code)

- 🌲 **Algorithm:** Random Forest Classifier (`scikit-learn`)
- 📂 **Data Split:**
  - Training set: **80%**
  - Testing set: **20%**

### Feature Importance Results

| Feature | Importance |
|------|------|
| Comorbidity Index | 49.1% |
| Age | 30.5% |
| Lab Procedures | 10.5% |

The model identified **comorbidity burden and age** as the strongest predictors.

---

## Phase 3: Statistical Validation (IBM SPSS)

To confirm the machine learning findings:

- 📥 Imported `readmission_analysis.csv` into SPSS
- 📊 Conducted **Binary Logistic Regression**
- 📈 Calculated **Odds Ratios (Exp(B))**
- 🔬 Performed **statistical significance testing**

### Key Result

Both major predictors achieved:
p-value < .001


Confirming they are **statistically significant predictors of readmission risk**.

---

## Phase 4: Visualization

### Python
- Feature importance **horizontal bar chart**

### SPSS
- **Clustered Bar Chart**
- Shows **readmission percentage across age groups**

This helps provide a **clear demographic perspective of risk patterns**.

---

# 📈 Project Summary

Key outcomes of this analysis include:

- 🎯 **Model Accuracy:** ~89–90%
- 🧠 **Top Predictor:** Comorbidity Index  
  - Odds Ratio: **20.773**
- 📊 **Statistical Significance:**  
  - Primary predictors with **p < .001**

This confirms that **patients with multiple chronic conditions and increasing age are at significantly higher risk of early readmission.**

---

# 🚀 Potential Applications

This model can help support:

- Hospital **discharge planning**
- **Patient risk stratification**
- Early **post-discharge interventions**
- **Healthcare policy planning**

---

# 📂 Repository Structure
Hospital-Readmission-Prediction/
│
├── data/
│ └── readmission_analysis.csv
│
├── notebooks/
│ └── model_training.ipynb
│
├── Analysis_Results/
│ └── Readmission_by_Age_Chart.png
│
├── spss_analysis/
│ └── logistic_regression_output.spv
│
└── README.md


---

# 👨‍💻 Author

Developed as a **Healthcare Data Analytics & Machine Learning project** integrating **Python-based predictive modeling** with **SPSS statistical validation**.

---

# 📜 License

This project is available for **educational and research purposes**.

