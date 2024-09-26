**Rock or Mine Prediction Using Sonar Dataset**

**Project Overview**

                         This project uses a machine learning model to classify objects as either rocks or mines based on sonar signal returns. The dataset consists of 60 features corresponding to the strength of sonar signals, and each sample is labeled as either rock (R) or mine (M).

**Dataset**
Source: UCI Machine Learning Repository (Sonar dataset)
Size: 208 samples
Features: 60 numerical features
Target:
R for Rocks
M for Mines

**Dataset Details**:
Each feature represents the energy value of a frequency component, obtained from sonar signals bounced off an object.
The target value indicates whether the object is a rock or a mine.

**Project Steps**:
1. Data Loading and Exploration:
Load the dataset.
Check for missing values, data types, and basic statistics of features.

2. Data Preprocessing:
Normalize or standardize the features to ensure uniform scaling.
Split the data into training and test sets for evaluation.

3. Model Selection:
Logistic Regression

4. Model Training and Evaluation:
Train the model using the training set.
Evaluate its performance using metrics such as accuracy

5. Final Model:
The final model with the best performance will be selected and saved for future predictions.

**Requirements**
Python 3.12
Libraries:
pandas
numpy
scikit-learn

Result 
The best model achieved an accuracy of 84% on the test set.