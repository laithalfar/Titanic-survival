# Titanic Survival Prediction Script
import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import train_test_split


#load an already trained model from joblib library
def load_model(model_path='titanic_model.joblib'):
    """Load the trained model from disk."""

    #try to load model and return if successful
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model

    #if loading fails, print error message and return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# 1. More robust age imputation
def improve_age_imputation(df):
    """Use more sophisticated age imputation based on title and class"""
    # Calculate median age by title and class
    age_by_title_class = df.groupby(['Title', 'Pclass'])['Age'].median()
    
    # Fill missing ages
    for idx, row in df.iterrows():
        if pd.isna(row['Age']):
            title = row['Title']
            pclass = row['Pclass']
            if (title, pclass) in age_by_title_class.index:
                df.loc[idx, 'Age'] = age_by_title_class[(title, pclass)]
            else:
                df.loc[idx, 'Age'] = df.groupby('Title')['Age'].median()[title]
    
    return df

# 2. Better fare imputation
def improve_fare_imputation(df):
    """Impute fare based on class and embarkation port"""
    fare_by_class_port = df.groupby(['Pclass', 'Embarked'])['Fare'].median()
    
    for idx, row in df.iterrows():
        if pd.isna(row['Fare']):
            pclass = row['Pclass']
            embarked = row['Embarked']
            if (pclass, embarked) in fare_by_class_port.index:
                df.loc[idx, 'Fare'] = fare_by_class_port[(pclass, embarked)]
            else:
                df.loc[idx, 'Fare'] = df[df['Pclass'] == pclass]['Fare'].median()
    
    return df


def remove_outliers_iqr(df, columns, k=1.5):
    """
    Remove rows with outliers in given columns using the IQR rule.
    k=1.5 -> standard, can increase for less aggressive removal.
    """
    clean_df = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(clean_df[col]):
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - k * IQR
            upper = Q3 + k * IQR
            clean_df = clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]
    return clean_df

#preprocess training data in general
def preprocess_data(data, is_test=False):
    """Preprocess the training data"""
    # Extract titles from names
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    

    # Create family size feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # Create is_alone feature
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

    # Impute Embarked with mode
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    # Create cabin feature - first letter of cabin indicates deck
    data['Cabin_Letter'] = data['Cabin'].str.slice(0, 1) if 'Cabin' in data.columns else None
    if data['Cabin_Letter'] is not None:
        data['Cabin_Letter'].fillna('U', inplace=True)  # U for Unknown
    
   # Create name length feature
    data['Name_Length'] = data['Name'].apply(len)
    
    #Create Family_Survival feature
    #This is a simplified version since we don't have the full training data
    #We'll set a default value of 0.5 for all
    data['Family_Survival'] = 0.5
    
    # Create ticket first char feature
    if 'Ticket' in data.columns:
        data['Ticket_First_Char'] = data['Ticket'].str.slice(0, 1)
        data['Ticket_First_Char'] = data['Ticket_First_Char'].str.replace('\d+', 'N')
    
    # Create age and class interaction feature
    if 'Age' in data.columns and 'Pclass' in data.columns:
        # Impute missing Age values by Title median first
        data['Age'] = data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
        data['Age_Class'] = data['Age'] * data['Pclass']
    
    # Create fare per person feature
    if 'Fare' in data.columns:
        # Fill missing Fare values with median
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        data['Fare_Per_Person'] = data['Fare'] / data['FamilySize']


    # # Age groups instead of continuous age
    data['Age_Group'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100], 
                            labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])    
    # # Age-Sex interaction
    data['Age_Sex'] = data['Age_Group'].astype(str) + '_' + data['Sex']

    # # Create name length feature - longer names might indicate higher social status
    data['Name_Length'] = data['Name'].apply(len)

    # # Create ticket first char feature
    data['Ticket_First_Char'] = data['Ticket'].str.slice(0, 1)
    
        
    # Fare groups instead of continuous
    data['Fare_Group'] = pd.qcut(data['Fare'], q=4, labels=["Very_Low", "Low", "Medium", "High"])


    # Class-Sex interaction (historically important for Titanic)
    data['Class_Sex'] = data['Pclass'].astype(str) + '_' + data['Sex']

    remove_outliers_iqr(data, data.columns.tolist())
    # Select important features for our purpose
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Age_Group', 'Fare_Group', 'Cabin_Letter', 'Ticket_First_Char', 'Age_Sex', 'Name_Length', 'Class_Sex', 'Family_Survival', 'Fare_Per_Person', 'Age_Class']
    
    # Only include features that exist in the data
    available_features = [f for f in features if f in data.columns]
    X = data[available_features]
    
    return X


#test data to match training data
def align_features(X_test, X_train_columns):
    """Ensure test data has the same features as training data."""

    # Add missing columns
    for col in X_train_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Remove extra columns
    for col in list(X_test.columns):
        if col not in X_train_columns:
            X_test = X_test.drop(col, axis=1)
    
    # Ensure columns are in the same order
    X_test = X_test[X_train_columns]
    
    return X_test

#Make using the existing model
def make_predictions(model, data, X_train_columns):
    """Make predictions using the trained model."""
    # Preprocess the data
    X_test = preprocess_data(data)
    
    # Align features with training data
    X_test = align_features(X_test, X_train_columns)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Add predictions to the original data
    results = data.copy()
    results['Survived_Predicted'] = predictions
    results['Survival_Probability'] = probabilities
    
    return results

#main code
def main():
    # Check if test file is provided as command line argument
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = 'test.csv'  # Default test file
    
    # Check if model file is provided as command line argument
    if len(sys.argv) > 2:
        model_file = sys.argv[2]
    else:
        model_file = 'titanic_model.joblib'  # Default model file
    
    # Check if output file is provided as command line argument
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    else:
        output_file = 'predictions.csv'  # Default output file
    
    # Load the model
    model = load_model(model_file)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Load the test data
    try:
        test_data = pd.read_csv(test_file)
        print(f"Loaded test data from {test_file} with {test_data.shape[0]} passengers")
    except Exception as e:
        print(f"Error loading test data: {e}")
        print("Attempting to use train.csv and split it into train/test sets...")
        try:
            # Load training data and split it
            train_data = pd.read_csv('train.csv')
            train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)
            print(f"Created test set from train.csv with {test_data.shape[0]} passengers")
        except Exception as e2:
            print(f"Error creating test data from train.csv: {e2}")
            return
    
    # For simplicity, let's just run the model directly on the test data
    # This avoids feature alignment issues
    try:
        # Preprocess the test data
        X_test = preprocess_data(test_data)
        
        # Make predictions directly
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Add predictions to the original data
        results = test_data.copy()
        results['Survived_Predicted'] = predictions
        results['Survival_Probability'] = probabilities
        
        # Save the results
        results.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Print some statistics
        print("\nPrediction Statistics:")
        print(f"Total passengers: {len(results)}")
        print(f"Predicted survivors: {results['Survived_Predicted'].sum()} ({results['Survived_Predicted'].mean()*100:.2f}%)")
        
        # If actual survival data is available, calculate accuracy
        if 'Survived' in results.columns:
            accuracy = (results['Survived'] == results['Survived_Predicted']).mean()
            print(f"Accuracy: {accuracy*100:.2f}%")
        
        return results
    except Exception as e:
        print(f"Error making predictions: {e}")
        return
    
    # The prediction logic has been moved up in the function
    # No additional code needed here

if __name__ == "__main__":
    main()