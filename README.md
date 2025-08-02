# Credit-Card-Approval-Predictor

This project builds a Machine Learning model to predict credit card approval probability. The data is processed using Pandas, visualized with Matplotlib/Seaborn, and then multiple classification models (such as Logistic Regression, KNN, XGBoost) are trained. The models are optimized using GridSearchCV to achieve the highest accuracy.

## Project Overview

The goal of this project is to build a classifier that can accurately predict whether a credit card application will be approved. The process involves several key stages:

1.  **Data Loading and Merging**: The project starts by loading two separate datasets—one with applicant information and another with the approval labels—and merging them into a single DataFrame.

2.  **Exploratory Data Analysis (EDA)**: To understand the data, various visualizations are created using Matplotlib and Seaborn. This includes analyzing the average income by gender, the distribution of credit card approvals, and the relationship between income and marital status.

3.  **Data Cleaning and Preprocessing**: The data is cleaned by handling missing values. The `Type_Occupation` column, having a significant number of null entries, is dropped, and any remaining rows with missing data are removed. Categorical features are converted into numerical format using `LabelEncoder`, and the data is scaled using `StandardScaler` to prepare it for modeling.

4.  **Model Training and Evaluation**: Several machine learning models are trained on the preprocessed data. The performance of each model is evaluated based on its accuracy score.

5.  **Hyperparameter Tuning**: To enhance model performance, `GridSearchCV` is used to find the optimal hyperparameters for each model. This systematic approach ensures that the models are fine-tuned for the best possible accuracy.

## Techniques and Technologies Used

* **Data Manipulation and Analysis**: **Pandas** for data loading, cleaning, and manipulation.
* **Data Visualization**: **Matplotlib** and **Seaborn** for creating plots and visualizations to explore the dataset.
* **Machine Learning**: **Scikit-learn** and **XGBoost** for building and training the classification models.
* **Models Implemented**:
    * Logistic Regression
    * K-Nearest Neighbors (KNN)
    * Support Vector Machine (SVM)
    * Decision Tree
    * Random Forest
    * AdaBoost
    * XGBoost
* **Model Optimization**: **GridSearchCV** for hyperparameter tuning to improve model accuracy.

## Results

After training and optimizing the various models, the project successfully identifies the best-performing models for predicting credit card approval. The optimized models consistently achieve a high accuracy of approximately **89%**, demonstrating their effectiveness in solving this classification problem.
