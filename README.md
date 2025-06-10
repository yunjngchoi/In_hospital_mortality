# Trauma In-Hospital Mortality Prediction Using ISS-Based AI Model

## **Project Overview**
This project evaluates the integration of patient demographic features (age and sex) into an Injury Severity Score (ISS)-based AI model to predict trauma-related in-hospital mortality accurately, fairly, and with generalizability across different clinical settings.

## **Objective**
The goal of this study is to assess whether scaling an ISS-based AI model with demographic features improves the predictive accuracy, fairness, and generalizability of trauma-related in-hospital mortality predictions.

## **Dataset**
- **Training and Internal Testing Data**: 121,418 trauma patients from 17 regional trauma centers in South Korea (2017–2022).
- **External Validation Data**: 7,458 trauma patients from five independent trauma centers (four in South Korea, one in Australia; 2022–2024).

## **Methods**
- **Design**: Retrospective multicenter, multinational cohort study.
- **Model Development**: AI-based model was trained using patient data and the Injury Severity Score (ISS) integrated with demographic factors (age, sex).
- **Evaluation**: The model's performance was assessed using sensitivity, specificity, accuracy, balanced accuracy, and AUROC. 
- **Fairness and Generalizability**: Comparison of AUROC differences across demographic subgroups (age, sex) for fairness, and external validation for generalizability.

## **Key Outcomes**
- **Primary Outcomes**: Predictive performance metrics for trauma-related in-hospital mortality, including:
  - Sensitivity, Specificity, Accuracy, Balanced accuracy, Area Under the Receiver Operating Characteristic Curve (AUROC)
  
- **Secondary Outcomes**:
  - Model fairness (assessed by comparing AUROC across demographic subgroups)
  - Generalizability (external validation across independent trauma centers)

## **Results**
- **Internal Testing AUROC**: 0.934
- **External Validation AUROC**: Range 0.902–0.920
- **Fairness**: The AI model demonstrated reduced prediction bias across demographic subgroups:
  - **Age**: AI model AUROC difference = 0.068 vs ISS = 0.091
  - **Sex**: AI model AUROC difference = 0.021 vs ISS = 0.046

## **Conclusions**
Integrating demographic features into an ISS-based AI model resulted in accurate, fair, and generalizable predictions of trauma-related in-hospital mortality. This model has the potential to significantly enhance clinical decision-making, optimize trauma care resource allocation, and improve patient outcomes globally.

## **File Structure**

1. **ais46_model_dnn.keras**: Contains the proposed DNN model for predicting trauma-related in-hospital mortality based on the ISS and demographic features (age and sex).

2. **data.py**: Handles loading, splitting, scaling, and oversampling of the dataset using MinMaxScaler and SMOTE for model training.

3. **model.py**: Defines and trains a DNN model with two hidden layers, evaluates the model using various metrics (accuracy, specificity, sensitivity, etc.), and applies early stopping during training.
