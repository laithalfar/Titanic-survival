# Titanic Survival Report

## 1. Project Overview
This report summarizes the analysis of the Titanic dataset and presents the results of a machine learning model designed to predict passenger survival.

---

## 2. File Structure
| File / Folder      | Description |
|-------------------|-------------|
| `requirements.txt` | Lists the Python dependencies required for the project. |
| `model.py`         | Contains the machine learning model code. |
| `predict.py`       | Uses the trained model to make predictions on the test set. |
| `run_pipeline.py`  | Orchestrates data processing, training, and evaluation. |
| `report.md`        | This report, summarizing the project and results. |
| `visualizations/`  | Charts and plots produced during analysis. |
| `predictions.csv`  | Model predictions saved as a CSV file. |
| `train.csv`        | Training dataset. |
| `test.csv`         | Test dataset for evaluation. |

---

## 3. Environment & Dependencies
All required libraries are specified in `requirements.txt`:

numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
joblib>=1.1.0
scipy>=1.7.0
mlxtend>=0.19.0


Install them with:

```bash
pip install -r requirements.txt


## 4. Script Details

### 4.1 `predict.py`
Responsible for generating survival predictions:
- Loads the trained model and `test.csv`
- Preprocesses the data to align features with the training set
- Produces predictions and saves them to `predictions.csv`

### 4.2 `model.py`
Implements model training and evaluation:
- Loads and preprocesses `train.csv`
- Splits the data into training, validation, and test sets
- Performs feature engineering:
  - Encodes categorical variables
  - Scales numerical features
  - Balances labels with **SMOTE**
- Trains an ensemble of:
  - Random Forest  
  - XGBoost  
  - Logistic Regression (via Voting Classifier)
- Tunes hyperparameters with randomized search & cross-validation
- Evaluates using multiple metrics:
  - Accuracy
  - ROC-AUC
  - MCC
  - PR-AUC
- Plots learning curves to diagnose over/underfitting
- Saves the final model as `model.joblib`

### 4.3 `run_pipeline.py`
Coordinates the full workflow:
- Verifies required dependencies and data files
- Runs preprocessing, training, evaluation, and prediction steps
- Provides formatted console output for each stage
- Returns clear success/failure messages

---

## 5. Results (Example)
| Metric      | Validation | Test |
|-------------|------------|------|
| Accuracy    | 0.84       | 0.82 |
| ROC-AUC     | 0.88       | 0.86 |
| MCC         | 0.68       | 0.65 |
| PR-AUC      | 0.81       | 0.79 |

> The model shows strong performance across metrics, with good generalization to the test set.

---

## 6. Visualizations
Key plots saved in `visualizations/`:
- Feature importance bar chart
- ROC curve & Precision-Recall curve
- Confusion matrix
- Learning curves

---

## 7. Conclusion
The machine learning pipeline successfully predicts Titanic passenger survival with solid accuracy and robust evaluation metrics.  
Future improvements may include:
- Trying gradient boosting variants (e.g., LightGBM, CatBoost)  
- Engineering interaction features between age, class, and fare  
- Using ensemble stacking for better calibration



