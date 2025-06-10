# **Trauma Mortality Prediction Using ISS-Based AI Model with Demographic Scaling**

## **📝 Overview**

This project presents a scalable and equitable AI model for predicting trauma-related in-hospital mortality by enhancing traditional **Injury Severity Score (ISS)** methods with demographic features—**age** and **sex**.  
The model demonstrates:

- High predictive accuracy  
- Reduced demographic bias  
- Strong generalizability across trauma centers in different countries  

This tool supports clinical decision-making and improves resource allocation in trauma care settings.

---

## **🏥 Study Setting & Dataset**

### **Design**
- Retrospective, multicenter, multinational cohort study

### **Development & Internal Testing**
- Data source: National trauma registry  
- Location: **17 regional trauma centers** in South Korea  
- Period: **2017–2022**

### **External Validation**
- Location: **5 independent trauma centers**  
  - 4 in South Korea  
  - 1 in Australia  
- Period: **2022–2024**

---

### **Participants**

#### Development Dataset:
- **121,418 trauma patients**
- Aged **15 years or older**
- From South Korea (2017–2022)

#### External Validation Dataset:
- **7,458 trauma patients**
- From independent trauma centers in South Korea and Australia (2022–2024)

---

## **🔍 Features**

### **Input Features**
- **Injury Severity Score (ISS)**
  - Summarizes injury severity from different body regions
- **Demographic Factors**
  - `Age` (continuous or categorized)
  - `Sex` (binary: Male/Female)

### **Model Outputs**
- Predicted probability of **in-hospital mortality**
- **Performance Metrics**:
  - AUROC, sensitivity, specificity, accuracy, balanced accuracy
- **Fairness Evaluation**:
  - AUROC differences across:
    - Age: `<65` vs `≥65`
    - Sex: Female vs Male
- **Generalizability**:
  - Validated on external datasets from independent trauma centers

---

## **📌 Keywords**
Artificial intelligence · Injury Severity Score · Trauma · Mortality · Demographic factors · Prediction model · Fairness · Generalizability
