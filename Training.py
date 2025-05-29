import csv 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read from csv file and store in dataframe
df = pd.read_csv("Downloads/train.csv")

#Create data frame variables x and y from csv columns; where 
#A data frame is a list of variables of 
#the same number of rows with unique row names
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

#converts categorical variables to dummy variables which means in this case
#it turns the Sex and Embarked column values into binary values (0 or 1)
#however the first column is dropped because you can't do multicollinearity
#if there are three columns given that binaries can only be compared in pairs.
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)


#create a new columns age missing and fill with 0s and 1s
df['Age_missing'] = df['Age'].isnull().astype(int)  # Do this *before* imputing if you want to use it

#extract title from name column
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
df['Title'] = df['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev', 'Jonkheer', 'Dona', 'Sir', 'Lady', 'Countess'], 'Rare')

#fill missing values in age column with median of age column of same title
df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

#familySize new columns filled with total number of family members
#gotten from the sibsp and parch columns + the member himself
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Bin age and fare
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 80])
df['FareBin'] = pd.cut(df['Fare'], bins=[0, 25, 50, 100, 250, 600])

# Compute survival rate matrix
heatmap_data = df.pivot_table(index='FareBin', columns='AgeBin', values='Survived', aggfunc='mean')
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, cmap='YlOrBr', aspect='auto', interpolation='nearest')

# Add labels
plt.xticks(ticks=np.arange(len(heatmap_data.columns)), labels=[str(c) for c in heatmap_data.columns], rotation=45)
plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=[str(r) for r in heatmap_data.index])
plt.xlabel('Age Group')
plt.ylabel('Fare Group')
plt.title('Survival Rate by Age and Fare')
plt.colorbar(label='Survival Rate')
plt.tight_layout()
plt.show()