# Biosignal-Activity-Detection
Erdos Data Science Boot Camp project

**Team member**

Tong Shan, Dushyanth Sirivolu, Fulya Tastan, Philip Barron, Larsen Linov, Ming Li
## Project Description
Our goal is to study the biosignal pattern of everyday activity like walking, running, lifting chairs, etc, and creates machine learning models to recognize human daily activities from biosignals recorded by wearable devices. These signals includes electrocardiography (ECG), electrodermal activity (EDA), and photoplethysmography (PPG), electromyography (EMG), wrist temperature (TEMP) and chest and wrist actigraphy (ACC). The algorithms can be used in detecting user's daily activities and monitoring user's health condition.
## Dataset
[ScientISST MOVE](https://doi.org/10.13026/sg89-qq52): Annotated Wearable Multimodal Biosignals recorded during Everyday Life Activities in Naturalistic Environments v1.0.0
- There were 17 study participants, ≈1 hour of data per person (sampling frequency=500 Hz).
- The sensors recorded 14 raw biosignals in 6 categories: Electrocardiography (ECG), Electrodermal activity (EDA), Photoplethysmography (PPG), Acceleration (ACC), Electromyography (EMG), Skin temperature (TMP).
- Corresponding to 8 activities: Baseline, Lift, Greetings, Gesticulate, Jumps, Walk_before, Run, Walk_after.
- The visualization of the dataset can be seen in [`Exploratory_Data_Analysis_Single_Participant.ipynb`](https://github.com/TongShan4869/Biosignal-Activity-Detection/blob/main/Exploratory_Data_Analysis_Single_Participant.ipynb)
## Stakeholders
- Wearable devices company
- Fitness app developer
- Clinical researchers / doctors
- Insurance company
- Wearable devices users
## Data processing
- Timepoints where the signal was missing or obviously incorrect were excluded.
- Continuous biosignal from each activity period in each subject was segmented into 10-second segments [`signal_segmentation.py`](https://github.com/TongShan4869/Biosignal-Activity-Detection/blob/main/signal_segmentation.py)
- Features from time domain, frequency domain and statistical metrics were extracted using existing packages ([BIOBSS](https://github.com/obss/BIOBSS), [BioSPPy](https://biosppy.readthedocs.io/en/stable/), [Neurokit2](https://github.com/neuropsychology/NeuroKit)) and custom functions.[`dataset_feature_extraction.py`](https://github.com/TongShan4869/Biosignal-Activity-Detection/blob/main/dataset_feature_extraction.py)
## Activity Classification ML Models [`./models`](https://github.com/TongShan4869/Biosignal-Activity-Detection/tree/main/models)
Hyperparameters were optimized through grid search by cross-validation in training set
- Random forest 
- Support vector machine (SVM)
- Logistic/softmax regression
- XGBoost
## Key resutls
The models all did well on classification on the test set, here are the log loss scores of each model and the confusion matrix from XGBoostClassifier. 

| Algorithms      | Log Loss Score |
| ----------- | ----------- |
| Logistic/SoftMax Regression      | 0.1263       |
| Support Vevtor Machine   | 0.1165        |
| Random Forest Classifier   | 0.1399        |
| XGBoost Classifier   | 0.0677       |

**Confusion Matrix of XGBoost Classifier**

<img width="660" alt="image1" src="https://github.com/TongShan4869/Biosignal-Activity-Detection/assets/51421789/38c60258-61a6-4ded-a877-48512714c861">


It is also found that the most important features in recognizing human activities come from the Acceleration (ACC) biosignal.

## Application
Our algorithms can be used for wearable devices to estimate user daily activities

## Reference
Areias Saraiva, J., Abreu, M., Carmo, A. S., Plácido da Silva, H., & Fred, A. (2023). ScientISST MOVE: Annotated Wearable Multimodal Biosignals recorded during Everyday Life Activities in Naturalistic Environments (version 1.0.0). PhysioNet. https://doi.org/10.13026/sg89-qq52.
