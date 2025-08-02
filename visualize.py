# Titanic Data and Model Visualization Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set style
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

def load_data(file_path='train.csv'):
    """Load the Titanic dataset."""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_visualization_folder():
    """Create a folder for visualizations if it doesn't exist."""
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
        print("Created 'visualizations' folder")

def visualize_survival_distribution(data):
    """Visualize the distribution of survival."""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Survived', data=data, palette='Set1')
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12)
    
    plt.title('Survival Distribution', fontsize=16)
    plt.xlabel('Survived (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig('visualizations/survival_distribution.png')
    plt.close()

def visualize_survival_by_feature(data, feature, title, xlabel):
    """Visualize survival rate by a categorical feature."""
    plt.figure(figsize=(12, 6))
    
    # Create a crosstab for percentage calculation
    survival_rate = pd.crosstab(data[feature], data['Survived'])
    survival_rate_pct = survival_rate.div(survival_rate.sum(axis=1), axis=0) * 100
    
    # Plot
    ax = survival_rate_pct[1].sort_values(ascending=False).plot(kind='bar', color='skyblue')
    
    # Add percentage labels
    for i, v in enumerate(survival_rate_pct[1].sort_values(ascending=False)):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.ylim(0, 100)  # Set y-axis to percentage scale
    plt.tight_layout()
    plt.savefig(f'visualizations/survival_by_{feature.lower()}.png')
    plt.close()

def visualize_age_distribution(data):
    """Visualize the age distribution by survival status."""
    plt.figure(figsize=(12, 6))
    
    # Create KDE plot
    sns.kdeplot(data=data, x='Age', hue='Survived', fill=True, common_norm=False, alpha=0.7, palette='Set1')
    
    plt.title('Age Distribution by Survival Status', fontsize=16)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.savefig('visualizations/age_distribution.png')
    plt.close()

def visualize_fare_distribution(data):
    """Visualize the fare distribution by survival status."""
    plt.figure(figsize=(12, 6))
    
    # Create boxplot
    sns.boxplot(x='Survived', y='Fare', data=data, palette='Set2')
    
    plt.title('Fare Distribution by Survival Status', fontsize=16)
    plt.xlabel('Survived (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Fare', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/fare_distribution.png')
    plt.close()

def visualize_correlation_matrix(data):
    """Visualize the correlation matrix of numerical features."""
    # Select numerical columns
    numerical_data = data.select_dtypes(include=['int64', 'float64'])
    
    # Calculate correlation matrix
    corr_matrix = numerical_data.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
    
    plt.title('Correlation Matrix of Numerical Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png')
    plt.close()

def visualize_survival_by_pclass_and_sex(data):
    """Visualize survival rate by passenger class and sex."""
    plt.figure(figsize=(12, 6))
    
    # Create crosstab
    survival_by_pclass_sex = pd.crosstab([data['Sex'], data['Pclass']], data['Survived'])
    survival_rate = survival_by_pclass_sex[1] / (survival_by_pclass_sex[0] + survival_by_pclass_sex[1]) * 100
    survival_rate = survival_rate.unstack()
    
    # Plot
    ax = survival_rate.plot(kind='bar', figsize=(12, 6))
    
    # Add percentage labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%')
    
    plt.title('Survival Rate by Passenger Class and Sex', fontsize=16)
    plt.xlabel('Sex', fontsize=12)
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.legend(title='Passenger Class')
    plt.ylim(0, 100)  # Set y-axis to percentage scale
    plt.tight_layout()
    plt.savefig('visualizations/survival_by_pclass_and_sex.png')
    plt.close()

def visualize_embarked_survival(data):
    """Visualize survival rate by port of embarkation."""
    plt.figure(figsize=(12, 6))
    
    # Create crosstab
    embarked_survival = pd.crosstab(data['Embarked'], data['Survived'])
    embarked_survival_rate = embarked_survival[1] / (embarked_survival[0] + embarked_survival[1]) * 100
    
    # Plot
    ax = embarked_survival_rate.plot(kind='bar', color='lightgreen')
    
    # Add percentage labels
    for i, v in enumerate(embarked_survival_rate):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.title('Survival Rate by Port of Embarkation', fontsize=16)
    plt.xlabel('Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)', fontsize=12)
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.ylim(0, 100)  # Set y-axis to percentage scale
    plt.tight_layout()
    plt.savefig('visualizations/survival_by_embarked.png')
    plt.close()

def visualize_family_size_survival(data):
    """Visualize survival rate by family size."""
    # Create family size feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    plt.figure(figsize=(14, 6))
    
    # Create crosstab
    family_survival = pd.crosstab(data['FamilySize'], data['Survived'])
    family_survival_rate = family_survival[1] / (family_survival[0] + family_survival[1]) * 100
    
    # Plot
    ax = family_survival_rate.plot(kind='bar', color='coral')
    
    # Add percentage labels
    for i, v in enumerate(family_survival_rate):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.title('Survival Rate by Family Size', fontsize=16)
    plt.xlabel('Family Size', fontsize=12)
    plt.ylabel('Survival Rate (%)', fontsize=12)
    plt.ylim(0, 100)  # Set y-axis to percentage scale
    plt.tight_layout()
    plt.savefig('visualizations/survival_by_family_size.png')
    plt.close()

def visualize_feature_importance(model_path='titanic_model.joblib'):
    """Visualize feature importance from the trained model."""
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Check if model is a pipeline
        if hasattr(model, 'steps'):
            # For sklearn pipeline, get the last step (classifier)
            classifier = model.steps[-1][1]
        else:
            # For standalone model
            classifier = model
        
        # Check if classifier has feature_importances_ attribute
        if hasattr(classifier, 'feature_importances_'):
            # Get feature importances
            importances = classifier.feature_importances_
            
            # Get feature names
            if hasattr(classifier, 'feature_names_in_'):
                feature_names = classifier.feature_names_in_
            else:
                # If feature names are not available, use generic names
                feature_names = [f'Feature {i}' for i in range(len(importances))]
            
            # Sort feature importances
            indices = np.argsort(importances)[::-1]
            sorted_feature_names = [feature_names[i] for i in indices]
            sorted_importances = importances[indices]
            
            # Plot
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
            plt.yticks(range(len(sorted_importances)), sorted_feature_names)
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance from Random Forest')
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png')
            plt.close()
            
            print("Feature importance visualization created")
        else:
            print("Model does not have feature_importances_ attribute")
    except Exception as e:
        print(f"Error visualizing feature importance: {e}")

def main():
    # Create visualization folder
    create_visualization_folder()
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_survival_distribution(data)
    visualize_survival_by_feature(data, 'Sex', 'Survival Rate by Sex', 'Sex')
    visualize_survival_by_feature(data, 'Pclass', 'Survival Rate by Passenger Class', 'Passenger Class')
    visualize_age_distribution(data)
    visualize_fare_distribution(data)
    visualize_correlation_matrix(data)
    visualize_survival_by_pclass_and_sex(data)
    visualize_embarked_survival(data)
    visualize_family_size_survival(data)
    
    # Visualize feature importance if model exists
    if os.path.exists('titanic_model.joblib'):
        visualize_feature_importance()
    
    print("All visualizations created successfully in the 'visualizations' folder")

if __name__ == "__main__":
    main()