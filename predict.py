# Titanic Survival Prediction Script
import pandas as pd
import numpy as np
import joblib
import sys
import os


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

#preprocess data in general
def preprocess_data(data):
    """Preprocess the test data in the same way as the training data."""
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
    
    # Select important features for our purpose
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
    X = data[features]
    
    # Cell 7: Impute missing Age values by Title median
    data['Age'] = data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, columns=['Sex', 'Embarked'])
    
    return X



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
        return
    
    # Get the training data columns (needed for feature alignment)
    # This assumes the model is a pipeline with the classifier as the last step

    #check for steps attribute to check the model is a pipeline
    if hasattr(model, 'steps'):

        # Accesses the last estimator or algorithm in the pipeline 
        # to check for an attribute only found in classifiers called 
        #feature_names_in
        if hasattr(model.steps[-1][1], 'feature_names_in_'):

            #if it has the attribute then it knows the feature name
            # and thus we can extyract the correct x train columns
            X_train_columns = model.steps[-1][1].feature_names_in_
        else:
            # If feature_names_in_ is not available, we need to infer from the training data
            print("Warning: Could not determine training features from model.")
            print("Please ensure test data has the same features as training data.")
            # Load training data to get columns
            try:
                train_data = pd.read_csv('train.csv')
                X_train = preprocess_data(train_data)
                X_train_columns = X_train.columns
            except Exception as e:
                print(f"Error loading training data: {e}")
                return
    else:
        # For standalone model
        if hasattr(model, 'feature_names_in_'):
            X_train_columns = model.feature_names_in_
        else:
            # If feature_names_in_ is not available, we need to infer from the training data
            print("Warning: Could not determine training features from model.")
            print("Please ensure test data has the same features as training data.")
            # Load training data to get columns
            try:
                train_data = pd.read_csv('train.csv')
                X_train = preprocess_data(train_data)
                X_train_columns = X_train.columns
            except Exception as e:
                print(f"Error loading training data: {e}")
                return
    
    # Make predictions
    results = make_predictions(model, test_data, X_train_columns)
    
    # Save predictions to CSV
    try:
        results.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"Total passengers: {results.shape[0]}")
    print(f"Predicted survivors: {results['Survived_Predicted'].sum()} ({results['Survived_Predicted'].mean()*100:.2f}%)")
    
    # If actual survival data is available, calculate accuracy
    if 'Survived' in results.columns:
        accuracy = (results['Survived'] == results['Survived_Predicted']).mean()
        print(f"Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()