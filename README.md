# Credit Card Approval Predictor

A Machine Learning project that predicts credit card approval probability using multiple classification algorithms. The project achieves **~89% accuracy** through systematic data preprocessing, feature engineering, and hyperparameter optimization.

## 📋 Project Summary

This project develops and compares multiple ML classifiers to predict whether a credit card application will be approved based on applicant information. The workflow includes:

- **Data Integration**: Merging applicant data with approval labels
- **Exploratory Data Analysis**: Understanding patterns in demographics, income, and approval rates
- **Data Preprocessing**: Handling missing values, encoding categorical features, and scaling
- **Model Training**: Implementing 7 different classification algorithms
- **Hyperparameter Tuning**: Optimizing models using GridSearchCV
- **Performance Evaluation**: Comparing models to identify the best predictor

## 🛠️ Tech Stack

### Core Libraries
- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations

### Data Visualization
- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical data visualization

### Machine Learning
- **Scikit-learn** - ML algorithms and preprocessing
  - `LabelEncoder` - Categorical encoding
  - `StandardScaler` - Feature scaling
  - `GridSearchCV` - Hyperparameter tuning
  - `train_test_split` - Data splitting
- **XGBoost** - Gradient boosting framework

### Development Environment
- **Jupyter Notebook** - Interactive development and analysis

## 🤖 Models Implemented

The following classification algorithms were trained and evaluated:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (SVM)**
4. **Decision Tree**
5. **Random Forest**
6. **AdaBoost**
7. **XGBoost**

## 📊 Model Performance

| Model | Accuracy (Before Tuning) | Accuracy (After GridSearchCV) |
|-------|-------------------------|-------------------------------|
| Logistic Regression | - | ~89% |
| KNN | - | ~89% |
| SVM | - | ~89% |
| Decision Tree | - | ~89% |
| Random Forest | - | ~89% |
| AdaBoost | - | ~89% |
| XGBoost | - | ~89% |

**Best Performance**: All optimized models achieve approximately **89% accuracy**, demonstrating robust predictive capability across different algorithm families.

## 🔄 Workflow

### 1. Data Loading and Merging
- Load `Credit_card.csv` (applicant information)
- Load `Credit_card_label.csv` (approval labels)
- Merge datasets into unified DataFrame

### 2. Exploratory Data Analysis (EDA)
- Analyze income distribution by gender
- Visualize approval rates
- Explore relationships between features (income, marital status, etc.)

### 3. Data Cleaning and Preprocessing
- Drop `Type_Occupation` column (high null rate)
- Remove remaining rows with missing values
- Encode categorical variables using `LabelEncoder`
- Scale features using `StandardScaler`

### 4. Model Training and Evaluation
- Split data into training and testing sets
- Train multiple classification models
- Evaluate performance using accuracy metrics

### 5. Hyperparameter Tuning
- Apply `GridSearchCV` to each model
- Find optimal hyperparameters
- Retrain models with best parameters

## 📁 Project Structure

```
Credit-Card-Approval-Predictor/
│
├── Credit_Card_Approval_Prediction.ipynb    # Main notebook with full analysis
├── Credit_card.csv                          # Applicant information dataset
├── Credit_card_label.csv                    # Approval labels
└── README.md                                # Project documentation
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/Freya-Ng/Credit-Card-Approval-Predictor.git
cd Credit-Card-Approval-Predictor
```

2. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook Credit_Card_Approval_Prediction.ipynb
```

4. Run all cells to reproduce the analysis and results.

## 📈 Key Insights

- Multiple ML algorithms perform consistently well on this dataset (~89% accuracy)
- Hyperparameter tuning with GridSearchCV ensures optimal model performance
- Proper preprocessing (encoding, scaling) is crucial for model effectiveness
- The dataset provides sufficient signal for reliable credit approval prediction

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Freya Ng**
- GitHub: [@Freya-Ng](https://github.com/Freya-Ng)
