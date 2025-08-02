# Titanic Survival Prediction

A machine learning project to predict passenger survival on the Titanic using various classification algorithms. This project analyzes the famous Titanic dataset to build a predictive model that determines which passengers survived the disaster.

## Project Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered "unsinkable" RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren't enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

In this project, we build a predictive model that answers the question: "What sorts of people were more likely to survive?" using passenger data (name, age, gender, socio-economic class, etc).

## Dataset

The data has been split into two groups:
- `train.csv` - Used to build the machine learning model
- `test.csv` - Used to test the model's predictions (not included in this repository)

### Features

- **PassengerId**: Unique identifier for each passenger
- **Survived**: Survival status (0 = No, 1 = Yes) - this is our target variable
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) - a proxy for socio-economic status
- **Name**: Passenger name
- **Sex**: Passenger sex
- **Age**: Passenger age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Project Structure

- `train.csv` - Training dataset
- `data_explore.ipynb` - Jupyter notebook for data exploration and analysis
- `model.py` - Script to train and evaluate the machine learning model
- `predict.py` - Script to make predictions on new data
- `requirements.txt` - List of required Python packages
- `README.md` - Project documentation

## Feature Engineering

The following features were engineered from the original dataset:

1. **Title** - Extracted from passenger names
2. **FamilySize** - Combination of siblings/spouses and parents/children
3. **IsAlone** - Binary indicator if a passenger is traveling alone
4. **AgeBand** - Age grouped into bands
5. **FareBand** - Fare grouped into bands

## Model

The project uses a Random Forest Classifier with hyperparameter tuning via GridSearchCV. The model is evaluated using:

- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix
- ROC curve and AUC
- Feature importance analysis

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model, run:

```
python model.py
```

This will:
- Load and preprocess the training data
- Train a Random Forest model with optimized hyperparameters
- Evaluate the model performance
- Save the trained model as `titanic_model.joblib`
- Generate visualization plots

### Making Predictions

To make predictions on new data, run:

```
python predict.py [test_file.csv] [model_file.joblib] [output_file.csv]
```

Arguments:
- `test_file.csv` - Path to the test data (default: 'test.csv')
- `model_file.joblib` - Path to the trained model (default: 'titanic_model.joblib')
- `output_file.csv` - Path to save predictions (default: 'predictions.csv')

## Results

The model achieves good performance on the training data with cross-validation. Key insights from the model:

- Gender is the most important feature for survival prediction
- Higher class passengers (1st class) had better survival rates
- Age played a significant role, with children having higher survival rates
- Family size affected survival chances

## Future Improvements

- Experiment with different algorithms (XGBoost, Neural Networks)
- More advanced feature engineering
- Ensemble methods combining multiple models
- Deep learning approaches for feature extraction

## License

This project is open source and available under the MIT License.
