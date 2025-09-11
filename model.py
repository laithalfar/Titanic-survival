# Titanic Survival Prediction Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, average_precision_score, precision_recall_curve, make_scorer, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from predict import preprocess_data
from joblib import dump
from collections import Counter
from sklearn.impute import SimpleImputer
import scipy.stats as stats  # allows you to define distributions
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer




# Set flag to indicate XGBoost is available
xgboost_available = True

# Load the dataset
df = pd.read_csv("train.csv")

# Data exploration
print("Dataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Feature engineering
X = preprocess_data(df)

#once preprocessing is applied check for missing values again
print("Missing values after preprocessing: " , df.isnull().sum())

"""Create interaction features that are less prone to overfitting"""

# Feature selection
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=432, stratify=y)

# Split a small validation set from training for internal evaluation
X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=432
)


#function for adding family_survival column
def create_family_survival_safe(df_train, df_all):
    """Calculate family survival only from training data to avoid leakage"""
    family_survival = {}
    
    # Calculate family survival rates only from training data
    for grp, grp_df in df_train.groupby(['Fare', 'Parch', 'SibSp']):
        if len(grp_df) > 1:
            survival_rate = grp_df['Survived'].mean()
            for _, row in grp_df.iterrows():
                # Use a family identifier (could be Fare + family size)
                family_id = f"{row['Fare']}_{row['Parch']}_{row['SibSp']}"
                family_survival[family_id] = survival_rate
    
    # Apply to all data
    df_all['Family_ID'] = df_all['Fare'].astype(str) + '_' + df_all['Parch'].astype(str) + '_' + df_all['SibSp'].astype(str)
    df_all['Family_Survival'] = df_all['Family_ID'].map(family_survival)
    df_all['Family_Survival'].fillna(0.5, inplace=True)  # Default for unknown families
    
    return df_all.drop('Family_ID', axis=1)

# Use the training indices to identify training data
train_indices = X_train_main.index
df_with_safe_family = create_family_survival_safe(df.iloc[train_indices], df.copy())

# NOW: Recreate your features for only training data with the corrected family survival
X_train_main = preprocess_data(df_with_safe_family.loc[train_indices])  # This should now include Family_Survival
X_val = preprocess_data(df_with_safe_family.loc[X_val.index])  # This should now include Family_Survival
X_test = preprocess_data(df_with_safe_family.loc[X_test.index])  # This should now include Family_Survival

# Fix potential data leakage in family survival feature

# Visualize survival rate by sex
plt.figure(figsize=(10, 6))
df_with_safe_family.groupby('Sex')['Survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Rate')
plt.savefig('survival_by_sex.png')

# Visualize survival rate by passenger class
plt.figure(figsize=(10, 6))
df_with_safe_family.groupby('Pclass')['Survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.savefig('survival_by_class.png')

# Visualize survival rate by family size
plt.figure(figsize=(10, 6))
df_with_safe_family.groupby('FamilySize')['Survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Family Size')
plt.ylabel('Survival Rate')
plt.savefig('survival_by_family.png')

# print the columns
print("Columns:", X.columns.tolist())
#analyze distribution shift before smote for better debugging after smote
def analyze_distribution_shift(X_train, X_test, feature_cols):
    """Check if test set has different distribution than training set"""
    shift_detected = []

    for col in feature_cols:
        # Ensure we are working with Series
        if isinstance(X_train[col], pd.DataFrame):
            continue  # Skip multi-column cases

        # Numeric columns (including one-hot)
        if pd.api.types.is_numeric_dtype(X_train[col]):
            ks_stat, p_value = stats.ks_2samp(X_train[col], X_test[col])
            if p_value < 0.05:
                shift_detected.append((col, 'continuous', ks_stat, p_value))

        # Only remaining non-numeric categorical columns
        else:
            train_counts = X_train[col].value_counts()
            test_counts = X_test[col].value_counts()

            # Align categories
            all_categories = set(train_counts.index) | set(test_counts.index)
            train_aligned = [train_counts.get(cat, 0) for cat in all_categories]
            test_aligned = [test_counts.get(cat, 0) for cat in all_categories]

            # Scale train counts to match test total
            total_train = sum(train_aligned)
            total_test = sum(test_aligned)
            if total_train > 0 and total_test > 0:
                train_scaled = [x * (total_test / total_train) for x in train_aligned]
                chi2_stat, p_value = stats.chisquare(f_obs=test_aligned, f_exp=train_scaled)
                if p_value < 0.05:
                    shift_detected.append((col, 'categorical', chi2_stat, p_value))
    
    return shift_detected

#prevent domain shift between test and train data
class DomainAdaptationScaler(BaseEstimator, TransformerMixin):
    """Scale features to reduce domain shift"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_scaled = X.copy()
        
        for col in X.columns:
            # Only scale numeric columns that are NOT boolean
            if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
                median = X[col].median()
                q75, q25 = X[col].quantile([0.75, 0.25])
                iqr = q75 - q25
                if iqr > 0:
                    X_scaled[col] = (X[col] - median) / iqr
                else:
                    X_scaled[col] = X[col] - median
            # else: skip booleans or categorical dummies
        return X_scaled

# Drop unnecessary columns
#X = X.drop(['AgeBand', 'FareBand'], axis=1, errors='ignore')


# identify categorical features to dummy variables
categorical_features = ['Sex', 'Embarked', 'FamilySize', 'IsAlone', 'Age_Group', 'Fare_Group', 'Cabin_Letter', 'Ticket_First_Char', 'Age_Sex', 'Name_Length', 'Class_Sex', 'Family_Survival', 'Fare_Per_Person', 'Age_Class']

#identify numerical features
numerical_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
]


# change categorical features into numerical and identify them with marks cat and num
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", "passthrough", numerical_features)  # keep numeric as-is
    ]
)

# Fit on train, transform both train & test while maintaining same column structure
X_train_main = preprocessor.fit_transform(X_train_main)
X_val = preprocessor.transform(X_val)  # avoid data leakage
X_test = preprocessor.transform(X_test)

# Get feature names after encoding done at lines 209-210
encoded_feature_names = preprocessor.get_feature_names_out()
X_train_main = pd.DataFrame(X_train_main, columns=encoded_feature_names, index=y_train_main.index)
X_val = pd.DataFrame(X_val, columns=encoded_feature_names, index=y_val.index)
X_test = pd.DataFrame(X_test, columns=encoded_feature_names, index=y_test.index)

# ADD HERE - Analyze if your test set is different from training
feature_cols = X_train_main.columns.tolist()
distribution_shifts_val = analyze_distribution_shift(X_train_main, X_val, feature_cols)
distribution_shifts_test = analyze_distribution_shift(X_train_main, X_test, feature_cols)

print(" DISTRIBUTION SHIFT ANALYSIS for val:")
if distribution_shifts_val:
    print("Features with significant distribution differences:")
    for feature, type_, stat, p_val in distribution_shifts_val:
        print(f"  • {feature} ({type_}): stat={stat:.4f}, p-value={p_val:.4f}")
else:
    print("No significant distribution shifts detected")


print(" DISTRIBUTION SHIFT ANALYSIS for test:")
if distribution_shifts_test:
    print("Features with significant distribution differences:")
    for feature, type_, stat, p_val in distribution_shifts_test:
        print(f"  • {feature} ({type_}): stat={stat:.4f}, p-value={p_val:.4f}")
else:
    print("No significant distribution shifts detected")

# Define a more robust cross-validation strategy
repeated_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=432)


# --- Scale features to reduce domain shift ---
domain_scaler = DomainAdaptationScaler()
X_train_scaled = domain_scaler.fit_transform(X_train_main)
X_val_scaled = domain_scaler.transform(X_val)
X_test_scaled = domain_scaler.transform(X_test)

# Check class distribution
print("Before SMOTE:", Counter(y_train))

# Store column names before imputation
train_columns = X_train_main.columns
test_columns = X_test.columns

# Impute missing values (after selection)
imputer = SimpleImputer(strategy='median')
X_train_for_smote = pd.DataFrame(imputer.fit_transform(X_train_main), columns=X_train_main.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# print the columns
print("Columns after imputing:", X.columns.tolist())

# Apply SMOTE
smote = SMOTE(random_state=432)
X_resampled, y_resampled = smote.fit_resample(X_train_for_smote, y_train_main)

print("After SMOTE:", Counter(y_resampled))

# ADD HERE - Analyze if your test set is different from training
feature_cols = X_train_main.columns.tolist()
distribution_shifts_val = analyze_distribution_shift(X_train_main, X_val, feature_cols)
distribution_shifts_test = analyze_distribution_shift(X_train_main, X_test, feature_cols)

print(" DISTRIBUTION SHIFT ANALYSIS:")
if distribution_shifts_val:
    print("Features with significant distribution differences:")
    for feature, type_, stat, p_val in distribution_shifts_val:
        print(f"  • {feature} ({type_}): stat={stat:.4f}, p-value={p_val:.4f}")
else:
    print("No significant distribution shifts detected")

print(" DISTRIBUTION SHIFT ANALYSIS:")
if distribution_shifts_test:
    print("Features with significant distribution differences:")
    for feature, type_, stat, p_val in distribution_shifts_test:
        print(f"  • {feature} ({type_}): stat={stat:.4f}, p-value={p_val:.4f}")
else:
    print("No significant distribution shifts detected")


# Base learners ( RF + XGBoost)
rf =  ('random_forest', RandomForestClassifier(
    n_estimators=100, max_depth=7, min_samples_split=15,
    min_samples_leaf=7, max_features='sqrt', random_state=432, class_weight = 'balanced'
))


# Define base learners
base_learners = [
    rf
]


# Count classes
counter = Counter(y_train)  # y_train is your target vector
num_positive = counter[1]   # Survived
num_negative = counter[0]   # Did not survive

xgb = ('xgb', XGBClassifier(
    n_estimators=150, learning_rate=0.05, max_depth=4,
    reg_alpha=0.5, reg_lambda=0.5, subsample=0.8,
    colsample_bytree=0.8, random_state=432, scale_pos_weight = (num_negative / num_positive)
))

# Add XGBoost if available
if xgboost_available:
    base_learners.append(xgb)
else:
    print("XGBoost not available, using only DecisionTree and RandomForest as base learners.")


# Meta learner (can still use Logistic Regression because it’s light & stable)
meta_learner = ('logistic_regression', LogisticRegression(
    C=0.1, penalty='l2', max_iter=1000, random_state=432, class_weight = 'balanced'
))

voting_clf = VotingClassifier(
    estimators=[
        ( rf),
        ( xgb),
        (meta_learner)
    ],
    voting='soft'  # Use probabilities
)

# Create a pipeline with preprocessing and model
pipeline = make_pipeline(
    StandardScaler(),
    voting_clf
    )


# Define hyperparameters for grid search using probability distributions
param_grid = {
     # Random Forest - more regularized
    'votingclassifier__random_forest__n_estimators': [50, 100, 150],  # Fewer trees
    'votingclassifier__random_forest__max_depth': [5, 7, 10],  # Shallower trees
    'votingclassifier__random_forest__min_samples_split': [10, 15, 20],  # More conservative splits
    'votingclassifier__random_forest__min_samples_leaf': [5, 7, 10],  # Larger leaf sizes
    'votingclassifier__random_forest__max_features': ['sqrt'],  # Feature subsampling
    
    # XGBoost - add more regularization
    'votingclassifier__xgb__n_estimators': [100, 150, 200],
    'votingclassifier__xgb__learning_rate': [0.01, 0.05, 0.1],  # Lower learning rates
    'votingclassifier__xgb__max_depth': [3, 4, 5],  # Shallower trees
    'votingclassifier__xgb__reg_alpha': [0.1, 0.5, 1.0],  # L1 regularization
    'votingclassifier__xgb__reg_lambda': [0.1, 0.5, 1.0],  # L2 regularization
    'votingclassifier__xgb__subsample': [0.8, 0.9],  # Row sampling
    'votingclassifier__xgb__colsample_bytree': [0.8, 0.9],  # Column sampling
    
    # Logistic Regression
    'votingclassifier__logistic_regression__C': [0.01, 0.1, 1.0, 10]
}


# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=100,  # increased number of random combinations to try
    cv=repeated_cv,
    scoring='recall',  # Changed to F1 score which balances precision and recall
    random_state=432,
    n_jobs=-1,
    verbose=1,
    return_train_score=True  # Return training scores for analysis
)

random_search.fit(X_train_main, y_train_main)

# Get the best model
best_model = random_search.best_estimator_
print("\nBest parameters:", random_search.best_params_)



# Define multiple scoring metrics for comprehensive evaluation
scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted',
    'roc_auc': 'roc_auc',
    'mcc': make_scorer(matthews_corrcoef)
}

# Perform cross-validation with multiple metrics
cv_results = cross_validate(best_model, X_train_main, y_train_main, 
                           cv=repeated_cv, scoring=scoring, 
                           return_train_score=True, n_jobs=-1)

# Print cross-validation results
print("\nCross-Validation Results (Mean ± Std):\n")
for metric in scoring.keys():
    test_metric = f"test_{metric}"
    train_metric = f"train_{metric}"
    print(f"{metric.capitalize()} (Test): {cv_results[test_metric].mean():.4f} ± {cv_results[test_metric].std():.4f}")
    print(f"{metric.capitalize()} (Train): {cv_results[train_metric].mean():.4f} ± {cv_results[train_metric].std():.4f}")
    print(f"Train-Test Gap: {cv_results[train_metric].mean() - cv_results[test_metric].mean():.4f}\n")


y_val_pred = best_model.predict(X_val)
y_val_proba = best_model.predict_proba(X_val)[:, 1]

# Calculate and print val set metrics
print("\nVal Set Metrics:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred, average="weighted"))
print("Recall:", recall_score(y_val, y_val_pred, average="weighted"))
print("F1 Score:", f1_score(y_val, y_val_pred, average="weighted"))
print("MCC:", matthews_corrcoef(y_val, y_val_pred))
print("ROC AUC:", roc_auc_score(y_val, y_val_proba) if len(np.unique(y_test)) == 2 else "N/A")
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
# Evaluate the model on the test set
best_model.fit(X_train_main, y_train_main)
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate and print test set metrics
print("\nTest Set Metrics:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_test_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_test_pred, average="weighted"))
print("MCC:", matthews_corrcoef(y_test, y_test_pred))
print("ROC AUC:", roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) == 2 else "N/A")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Print confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not Survived', 'Survived'])
plt.yticks(tick_marks, ['Not Survived', 'Survived'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')

# Calculate ROC curve and AUC
y_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
roc_auc = auc(fpr, tpr)


print("True Positives (TP):", tp)
print("False Positives (FP):", fp)
print("True Negatives (TN):", tn)
print("False Negatives (FN):", fn)


 
# Balanced Accuracy
cv_balanced = cross_val_score(best_model, X_train_main, y_train_main, cv=repeated_cv, scoring='balanced_accuracy')
print(f"Cross-Validation Balanced Accuracy: {cv_balanced.mean():.4f} ± {cv_balanced.std():.4f}")
# MCC - using make_scorer with matthews_corrcoef

mcc_scorer = make_scorer(matthews_corrcoef)
cv_mcc = cross_val_score(best_model, X_train_main, y_train_main, cv=repeated_cv, scoring=mcc_scorer)
print(f"Cross-Validation MCC: {cv_mcc.mean():.4f} ± {cv_mcc.std():.4f}")

# Calculate PR AUC (average precision)
pr_auc = average_precision_score(y_test, y_proba)
print(f"PR AUC: {pr_auc:.4f}")



# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

# Advanced model evaluation techniques

# 1. Learning curves to diagnose overfitting/underfitting
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))
    plt.title('Learning Curves')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1_weighted')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.legend(loc='best')
    plt.savefig('learning_curve.png')
    return plt

# Plot learning curve
print("\nGenerating learning curves...")
plot_learning_curve(best_model, X_resampled, y_resampled)

# 2. Permutation feature importance (more reliable than model-based importance)


# Calculate permutation importance
print("\nCalculating permutation feature importance...")
perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Sort features by importance
feature_names = X_test.columns
perm_sorted_idx = perm_importance.importances_mean.argsort()[::-1]

# Plot permutation importance
plt.figure(figsize=(12, 8))
plt.barh(range(len(perm_sorted_idx)), perm_importance.importances_mean[perm_sorted_idx])
plt.yticks(range(len(perm_sorted_idx)), [feature_names[i] for i in perm_sorted_idx])
plt.xlabel('Permutation Importance')
plt.title('Permutation Feature Importance')
plt.tight_layout()
plt.savefig('permutation_importance.png')

# Print permutation importance
print("\nPermutation Feature Importances:")
for i in perm_sorted_idx:
    print(f"{feature_names[i]}: {perm_importance.importances_mean[i]:.4f} ± {perm_importance.importances_std[i]:.4f}")

# 3. Model-based feature importance (if available)
if hasattr(best_model[-1], 'feature_importances_'):
    # Get feature importances
    importances = best_model[-1].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.title('Model-based Feature Importances')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Print
    print("\nModel-based Feature Importances:")
    for i in range(len(indices)):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Save the model for later use
dump(best_model, 'titanic_model.joblib')
print("\nModel saved as 'titanic_model.joblib'")

# Function to predict survival for new passengers
def predict_survival(passenger_data):
    """Predict survival for new passenger data.
    
    Args:
        passenger_data: DataFrame with passenger information
        
    Returns:
        Survival predictions (0 or 1)
    """
    # Preprocess the data (same as training data)
    passenger_data['Title'] = passenger_data['Name'].str.extract(' ([A-Za-z]+)\.',
     expand=False)
    passenger_data['Title'] = passenger_data['Title'].replace(['Lady', 'Countess',
    'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    passenger_data['Title'] = passenger_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    passenger_data['Title'] = passenger_data['Title'].replace('Mme', 'Mrs')
    passenger_data['FamilySize'] = passenger_data['SibSp'] + passenger_data['Parch'] + 1
    passenger_data['IsAlone'] = 0
    passenger_data.loc[passenger_data['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
    'Title', 'FamilySize', 'IsAlone']
    X_new = passenger_data[features]
    
    # Convert categorical variables
    X_new = pd.get_dummies(X_new, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    
    # Ensure all columns from training are present
    for col in X.columns:
        if col not in X_new.columns:
            X_new[col] = 0
    
    # Ensure columns are in the same order
    X_new = X_new[X.columns]
    
    # Make predictions
    return best_model.predict(X_new)

print("\nModel training and evaluation complete!")