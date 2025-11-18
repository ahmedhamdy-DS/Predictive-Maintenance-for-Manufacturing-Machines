# Predictive Maintenance for Manufacturing Machines

##  Project Overview
This project focuses on **predicting machine failures** using sensor data from manufacturing machines.  
The goal is to provide **actionable insights** to reduce downtime, improve maintenance planning, and save costs.

---

##  Problem Statement
- Predict whether a machine will **fail in the near future** (**binary classification**)  
- Identify the **type of failure** for actionable maintenance planning (**multi-class classification**)  
- Understand **key features influencing machine failure**, like temperature, torque, rotational speed, and tool wear

---

## 📊 Data Description
The dataset contains **10,000 rows** with **14 features**, including:  
- Sensor readings: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`  
- Product type: `Type` (L, M, H)  
- Target:  
  - `Target` → 0 = Normal, 1 = Failure  
  - `Failure Type` → Heat Dissipation, Power, Overstrain, etc.  

**Example Table:**
| Feature | Description |
|---------|-------------|
| Air temperature [K] | Ambient temperature |
| Tool wear [min] | Time machine has been operating before maintenance |
| Rotational speed [rpm] | Speed of the motor |

![Machine Sensors Overview](images/machine_sensors_summary.png)

---

##  Exploratory Data Analysis (EDA)
Key insights:  
- **Target Distribution:** Only ~3.4% of events are failures → dataset is highly imbalanced  
- **Failure Types:** Most frequent: `Heat Dissipation Failure` & `Power Failure`  
- **Feature Distributions:**  
  - Outliers detected in `Torque` and `Rotational speed`  
  - Tool wear is the strongest predictor of failure  
- **Correlations:**  
  - Negative correlation between `Rotational speed` and `Torque`  
  - Failures cluster in **Low Speed/High Torque** and **High Speed/Low Torque** regions  

**Visualizations:**  
- Distribution plots, boxplots, heatmaps  
- Failure clusters scatter plots  
- Failure rate by product type  

---

##  Models & Training
**Preprocessing Pipeline:**  
- Numeric features → median imputation + standard scaling  
- Categorical features → most frequent imputation + OneHotEncoding  

**Models Tested:**  
- Logistic Regression  
- Random Forest, ExtraTrees  
- GradientBoosting, AdaBoost  
- XGBoost, LightGBM, CatBoost  

**Evaluation Metrics:**  
- ROC-AUC, F1 Score (Macro), Recall, Precision  
- ROC-AUC curves for model comparison  

**Key Results:**  
| Model | AUC | F1 (Macro) | Recall (Macro) | Precision (Macro) |
|-------|-----|------------|----------------|------------------|
| XGBoost | 0.978 | 0.872 | 0.882 | 0.862 |
| LightGBM | 0.978 | 0.841 | 0.875 | 0.813 |
| GradientBoosting | 0.977 | 0.855 | 0.798 | 0.942 |
| CatBoost | 0.977 | 0.850 | 0.862 | 0.839 |

---

## 💡 Insights & Interpretation
- **Tool wear [min]** is the strongest predictor of failure  
- **Torque [Nm]** and **Rotational speed [rpm]** also influence failure probability  
- Failures occur mostly in specific operating conditions → **high-strain zones**  
- **Product type L** has the highest failure rate proportionally, despite being the most common  
- Predictive models allow **maintenance teams to schedule interventions proactively**  

---

##  Potential Solutions
1. **Predictive Maintenance:**  
   - Schedule interventions before failures occur based on predicted probability  
   - Focus on machines at risk of Heat Dissipation or Power Failure  

2. **Maintenance Prioritization:**  
   - Use predicted failure types to allocate maintenance resources efficiently  

3. **Operational Guidelines:**  
   - Avoid low speed/high torque or high speed/low torque zones for prolonged periods  

4. **Further Data Collection:**  
   - Include additional machine sensors or environmental data to improve accuracy  

---

##  Requirements / Dependencies
See `requirements.txt` for full package list

---

## 🚀 How to Run
1. Clone the repository:  
```bash
git clone https://github.com/yourusername/manufacturing-predictive-maintenance.git
