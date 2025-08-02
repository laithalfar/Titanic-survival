# Titanic Survival Prediction Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from predict import preprocess_data

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
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

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
roc_auc = auc(fpr, tpr)

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

# Feature importance
if hasattr(best_model[-1], 'feature_importances_'):
    importances = best_model[-1].feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X_train.columns
    
    #visualize
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    #print
    print("\nFeature Importances:")
    for i in range(len(indices)):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Save the model
from joblib import dump
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