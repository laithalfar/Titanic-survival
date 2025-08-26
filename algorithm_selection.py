# Titanic Survival Prediction Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from predict import preprocess_data
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load the dataset
df = pd.read_csv("train.csv")

# Data exploration
print("Dataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Feature engineering
X = preprocess_data(df)# Create age bands
X['AgeBand'] = pd.cut(df['Age'], 5)

# # Create fare bands
X['FareBand'] = pd.cut(df['Fare'], 4)

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


original_feature_names = X.columns

# Fill missing values
imputer = SimpleImputer(strategy='median')  # You can use 'mean', 'most_frequent', etc.
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


X_train = pd.DataFrame(X_train, columns=original_feature_names)
X_test = pd.DataFrame(X_test, columns=original_feature_names)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE:", Counter(y_resampled))

MODEL_NAME = "decision_tree"
MODEL = DecisionTreeClassifier(random_state=42)

# Create a pipeline with preprocessing and model
pipeline = make_pipeline(
    StandardScaler(),
    DecisionTreeClassifier(random_state=42)
)


# Define hyperparameters for grid search
param_grid = {
    'decisiontreeclassifier__max_depth': [3, 5, 7, None],
    'decisiontreeclassifier__min_samples_split': [2, 5, 10],
    'decisiontreeclassifier__min_samples_leaf': [1, 2, 4],
    'decisiontreeclassifier__criterion': ['gini', 'entropy']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)

# Get the best model
best_model = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"DECISION TREE MODEL STATS")
print(f"\nAccuracy: {accuracy:.4f}")
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Calculate ROC curve and AUC
y_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)


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

print("\nModel training and evaluation complete for decision tree algo!")

MODEL_NAME = "random_forest"
MODEL = RandomForestClassifier(random_state=42)
# Create a pipeline with preprocessing and model
pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)


# Define hyperparameters for grid search
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_depth': [None, 5, 10],
    'randomforestclassifier__min_samples_split': [2, 5],
    'randomforestclassifier__min_samples_leaf': [1, 2]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)

# Get the best model
best_model = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"DECISION TREE MODEL STATS")
print(f"\nAccuracy: {accuracy:.4f}")
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Calculate ROC curve and AUC
y_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)


# Save the model for later use
dump(best_model, 'titanic_model.joblib')
print("\nModel saved as 'titanic_model.joblib'")
print("\nModel training and evaluation complete!")

MODEL_NAME = "AdaBoost"
MODEL = AdaBoostClassifier(random_state=42)
# Create a pipeline with preprocessing and model
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    AdaBoostClassifier(random_state=42)
)

param_grid = {
    'adaboostclassifier__n_estimators': [50, 100, 200],
    'adaboostclassifier__learning_rate': [0.01, 0.1, 1]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)

# Get the best model
best_model = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Adaboost MODEL STATS")
print(f"\nAccuracy: {accuracy:.4f}")
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Calculate ROC curve and AUC
y_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)


# Save the model for later use
dump(best_model, 'titanic_model.joblib')
print("\nModel saved as 'titanic_model.joblib'")
print("\nModel training and evaluation complete!")