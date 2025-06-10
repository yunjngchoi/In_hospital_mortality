# Trauma In-Hospital Mortality Prediction Using ISS-Based AI Model

## **Overview**
Integrating demographic features into an ISS-based AI model resulted in accurate, fair, and generalizable predictions of trauma-related in-hospital mortality. This model has the potential to significantly enhance clinical decision-making, optimize trauma care resource allocation, and improve patient outcomes globally.

## **Dataset**
- **Training and Internal Testing Data**: 121,418 trauma patients from 17 regional trauma centers in South Korea (2017–2022).
- **External Validation Data**: 7,458 trauma patients from five independent trauma centers (four in South Korea, one in Australia; 2022–2024).

## **Methods**
- **Design**: Retrospective multicenter, multinational cohort study.
- **Model Development**: AI-based model was trained using patient data and the Injury Severity Score (ISS) integrated with demographic factors (age, sex).
- **Evaluation**: The model's performance was assessed using sensitivity, specificity, accuracy, balanced accuracy, and AUROC. 
- **Fairness and Generalizability**: Comparison of AUROC differences across demographic subgroups (age, sex) for fairness, and external validation for generalizability.

## **Results**
- **Internal Testing AUROC**: 0.934
- **External Validation AUROC**: Range 0.902–0.920
- **Fairness**: The AI model demonstrated reduced prediction bias across demographic subgroups:
  - **Age**: AI model AUROC difference = 0.068 vs ISS = 0.091
  - **Sex**: AI model AUROC difference = 0.021 vs ISS = 0.046

## **File Structure**

- ais46_model_dnn.keras : Our proposed DNN model for predicting in-hospital mortality based on the Region-46 and demographic features (age and sex).
- data.py : Handles data loading and preprocessing for internal and external validation datasets.
- model.py : Defines and trains DNN model with two hidden layers, evaluates model's performance.
- README.md : Provides an overview of the project and usage instructions.
