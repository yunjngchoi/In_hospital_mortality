# **Trauma Mortality Prediction Using ISS-Based AI Model with Demographic Scaling**

## **Overview**

This project presents a scalable and equitable AI model for predicting trauma-related in-hospital mortality by enhancing traditional **Injury Severity Score (ISS)** methods with demographic features—**age** and **sex**.  
The model demonstrates:

- High predictive accuracy  
- Reduced demographic bias  
- Strong generalizability across trauma centers in different countries  

This tool supports clinical decision-making and improves resource allocation in trauma care settings.

---

## **Study Setting & Dataset**

### **Design**
- Retrospective, multicenter, multinational cohort study

### **Development & Internal Validation**
- **121,418 trauma patients**
- From South Korea (2017–2022)

### **External Validation**
- **7,458 trauma patients**
- From independent trauma centers in South Korea and Australia (2022–2024)

---

## **Features**

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
