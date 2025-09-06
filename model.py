# Titanic Survival Prediction Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, average_precision_score, precision_recall_curve, make_scorer, roc_auc_score
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from predict import preprocess_data
from joblib import dump
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
import scipy.stats as stats  # allows you to define distributions
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance


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

# Create age bands with more granular bins
X['AgeBand'] = pd.cut(df['Age'], 8)

# Create fare bands with more granular bins
X['FareBand'] = pd.cut(df['Fare'], 6)

# Create cabin feature - first letter of cabin indicates deck
df['Cabin_Letter'] = df['Cabin'].str.slice(0, 1)
df['Cabin_Letter'].fillna('U', inplace=True)  # U for Unknown
X['Cabin_Letter'] = df['Cabin_Letter']

# Create name length feature - longer names might indicate higher social status
X['Name_Length'] = df['Name'].apply(len)

# Create ticket first char feature
X['Ticket_First_Char'] = df['Ticket'].str.slice(0, 1)
X['Ticket_First_Char'] = X['Ticket_First_Char'].str.replace('\d+', 'N')

# Create age and class interaction feature
X['Age_Class'] = X['Age'] * df['Pclass']

# Create fare per person feature
X['Fare_Per_Person'] = df['Fare'] / X['FamilySize']

# Fix potential data leakage in family survival feature
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


# Visualize survival rate by sex
plt.figure(figsize=(10, 6))
df.groupby('Sex')['Survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Rate')
plt.savefig('survival_by_sex.png')

# Visualize survival rate by passenger class
plt.figure(figsize=(10, 6))
df.groupby('Pclass')['Survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.savefig('survival_by_class.png')

# Visualize survival rate by family size
plt.figure(figsize=(10, 6))
df.groupby('FamilySize')['Survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Family Size')
plt.ylabel('Survival Rate')
plt.savefig('survival_by_family.png')

# Feature selection
y = df['Survived']

# Drop unnecessary columns
X = X.drop(['AgeBand', 'FareBand'], axis=1, errors='ignore')

# Convert categorical features to dummy variables
categorical_features = ['Sex', 'Embarked', 'Cabin_Letter', 'Ticket_First_Char', 'Title']
for feature in categorical_features:
    if feature in X.columns:
        dummies = pd.get_dummies(X[feature], prefix=feature, drop_first=True)
        X = pd.concat([X.drop(feature, axis=1), dummies], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Implement feature selection using SelectFromModel with RandomForest
print("\nImplementing feature selection using SelectFromModel...")

# Create a temporary RandomForest model for feature selection
feature_selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'
)

# Store original feature names
original_feature_names = X_train.columns.tolist()

# Fit the selector to the training data
feature_selector.fit(X_train, y_train)

# Get selected feature mask and names
selected_features_mask = feature_selector.get_support()
selected_feature_names = [original_feature_names[i] for i in range(len(original_feature_names)) if selected_features_mask[i]]

print(f"Selected {len(selected_feature_names)} features out of {len(original_feature_names)}")
print("Selected features:", selected_feature_names)

# Transform the data to include only selected features
X_train_selected = feature_selector.transform(X_train)
X_test_selected = feature_selector.transform(X_test)

# Convert back to DataFrame with selected feature names
X_train = pd.DataFrame(X_train_selected, columns=selected_feature_names)
X_test = pd.DataFrame(X_test_selected, columns=selected_feature_names)

# Visualize selected vs. all features
plt.figure(figsize=(10, 6))
plt.bar(range(len(selected_features_mask)), selected_features_mask.astype(int))
plt.xticks(range(len(selected_features_mask)), original_feature_names, rotation=90)
plt.title('Selected Features')
plt.tight_layout()
plt.savefig('feature_selection.png')

# Check class distribution
print("Before SMOTE:", Counter(y_train))

# Store column names before imputation
train_columns = X_train.columns
test_columns = X_test.columns

# Fill missing values
imputer = SimpleImputer(strategy='median')  # You can use 'mean', 'most_frequent', etc.
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Convert back to DataFrame with the stored column names
X_train = pd.DataFrame(X_train, columns=train_columns)
X_test = pd.DataFrame(X_test, columns=test_columns)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE:", Counter(y_resampled))

# Base learners (DecisionTree + RF + XGBoost)

# Define base learners
base_learners = [
    ('decision_tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42))
]

# Add XGBoost if available
if xgboost_available:
    base_learners.append(('xgb', XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=1.5
    )))
else:
    print("XGBoost not available, using only DecisionTree and RandomForest as base learners.")


# Meta learner (can still use Logistic Regression because it’s light & stable)
meta_learner = LogisticRegression(max_iter=1000)

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(class_weight="balanced", random_state=42),
    cv=5,
    n_jobs=-1
)
# Create a pipeline with preprocessing and model
pipeline = make_pipeline(
    StandardScaler(),
    stacking_clf
    )


# Define hyperparameters for grid search using probability distributions
param_grid = {
    # Decision Tree parameters
    'stackingclassifier__decision_tree__max_depth': [3, 5, 7, 9, None],
    'stackingclassifier__decision_tree__min_samples_split': stats.randint(2, 20),
    'stackingclassifier__decision_tree__min_samples_leaf': stats.randint(1, 10),
    'stackingclassifier__decision_tree__criterion': ['gini', 'entropy'],
    'stackingclassifier__decision_tree__max_features': ['sqrt', 'log2', None],
    
    # Random Forest parameters
    'stackingclassifier__random_forest__n_estimators': stats.randint(100, 500),
    'stackingclassifier__random_forest__max_depth': [None, 5, 10, 15, 20],
    'stackingclassifier__random_forest__min_samples_split': stats.randint(2, 20),
    'stackingclassifier__random_forest__min_samples_leaf': stats.randint(1, 10),
    'stackingclassifier__random_forest__max_features': ['sqrt', 'log2', None],
    'stackingclassifier__random_forest__bootstrap': [True, False],
    'stackingclassifier__random_forest__class_weight': ['balanced', 'balanced_subsample', None],
    
    # XGBoost parameters (only included if XGBoost is available)
    'stackingclassifier__xgb__n_estimators': stats.randint(100, 500),
    'stackingclassifier__xgb__learning_rate': stats.uniform(0.01, 0.2),
    'stackingclassifier__xgb__max_depth': stats.randint(3, 8),
    'stackingclassifier__xgb__subsample': stats.uniform(0.6, 0.4),
    'stackingclassifier__xgb__colsample_bytree': stats.uniform(0.6, 0.4),
    'stackingclassifier__xgb__reg_alpha': stats.uniform(0, 1),
    'stackingclassifier__xgb__reg_lambda': stats.uniform(0, 1),
    'stackingclassifier__xgb__scale_pos_weight': stats.uniform(1, 3),
    
    # Meta-learner (LogisticRegression) parameters
    'stackingclassifier__final_estimator__C': stats.uniform(0.1, 10),
    'stackingclassifier__final_estimator__solver': ['liblinear', 'saga'],
    'stackingclassifier__final_estimator__penalty': ['l1', 'l2']
}


# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=100,  # increased number of random combinations to try
    cv=5,
    scoring='f1',  # Changed to F1 score which balances precision and recall
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=True  # Return training scores for analysis
)
random_search.fit(X_resampled, y_resampled)

# Get the best model
best_model = random_search.best_estimator_
print("\nBest parameters:", random_search.best_params_)


# Define a more robust cross-validation strategy
repeated_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

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

cv_results = cross_validate(best_model, X_resampled, y_resampled, 
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

# Evaluate the model on the test set
best_model.fit(X_resampled, y_resampled)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate and print test set metrics
print("\nTest Set Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
print("MCC:", matthews_corrcoef(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else "N/A")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
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
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
roc_auc = auc(fpr, tpr)


print("True Positives (TP):", tp)
print("False Positives (FP):", fp)
print("True Negatives (TN):", tn)
print("False Negatives (FN):", fn)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Balanced Accuracy
cv_balanced = cross_val_score(best_model, X_resampled, y_resampled, cv=cv, scoring='balanced_accuracy')
print(f"Cross-Validation Balanced Accuracy: {cv_balanced.mean():.4f} ± {cv_balanced.std():.4f}")
# MCC - using make_scorer with matthews_corrcoef

mcc_scorer = make_scorer(matthews_corrcoef)
cv_mcc = cross_val_score(best_model, X_resampled, y_resampled, cv=cv, scoring=mcc_scorer)
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