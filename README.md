Trauma Mortality Prediction Using ISS-Based AI Model with Demographic Scaling
Overview
This project presents a scalable and equitable AI model for predicting trauma-related in-hospital mortality by enhancing traditional Injury Severity Score (ISS) methods with demographic features—age and sex. The model demonstrates high predictive accuracy, reduced demographic bias, and strong generalizability across independent trauma centers in different countries. It has been developed to support clinical decision-making and improve resource allocation in trauma care settings.

Study Setting & Dataset
Design: Retrospective, multicenter, multinational cohort study

Development & Internal Testing: Conducted using a nationwide trauma registry dataset collected from 17 regional trauma centers in South Korea, spanning 2017–2022

External Validation: Performed on data from 5 independent trauma centers (4 in South Korea and 1 in Australia), covering the period 2022–2024

Participants
Development Dataset:

121,418 trauma patients

Aged 15 years or older

South Korea (2017–2022)

External Validation Dataset:

7,458 trauma patients

From independent centers in South Korea and Australia (2022–2024)

Features
The model utilizes the following input features:

Injury Severity Score (ISS)

A traditional summary measure based on the most severe injuries in different body regions

Demographic Features

Age (continuous or categorical)

Sex (binary)

Model Outputs

Predicted probability of in-hospital mortality

Performance metrics: AUROC, sensitivity, specificity, accuracy, balanced accuracy

Fairness metrics: AUROC difference across age and sex subgroups

Generalizability performance on external validation datasets

