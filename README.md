# Trauma In-Hospital Mortality Prediction Using ISS-Based AI Model

## **Overview**
Integrating demographic features into an ISS-based AI model resulted in accurate, fair, and generalizable predictions of trauma-related in-hospital mortality. This model has the potential to significantly enhance clinical decision-making, optimize trauma care resource allocation, and improve patient outcomes globally.

## **Dataset**
**Development & Internal Validation**
- 121,418 trauma patients from 17 regional trauma centers in South Korea (2017–2022).
  
**External Validation**
- 7,458 trauma patients from five independent trauma centers (4 in South Korea, 1 in Australia; 2022–2024).

## **Methods**
**Design**
- Retrospective multicenter, multinational cohort study.
**Development**
- model was trained using Injury Severity Score (Region-1999, Region-6, Region-46) and demographic factors (age, sex).
**Evaluation**
- model's performance was assessed using sensitivity, specificity, accuracy, balanced accuracy, and AUROC. 
**Fairness and Generalizability**
- Comparison of AUROC differences across demographic subgroups for fairness, and external validation for generalizability.

## **Results**
- Internal Validation AUROC : 0.934
- External Validation AUROC : 0.902–0.920
- Fairness AUROC difference : Age (our model = 0.068 vs ISS = 0.091), Sex (our model = 0.021 vs ISS = 0.046)

## **File Structure**
- `ais46_model_dnn.keras`: Our proposed DNN model for predicting in-hospital mortality based on Region-46 and demographic features.
- `data.py` : Handles data loading and preprocessing for internal and external validation datasets.
- `model.py` : Defines and trains DNN model with two hidden layers, evaluates model's performance.
- `README.md` : Provides an overview of the project and usage instructions.
