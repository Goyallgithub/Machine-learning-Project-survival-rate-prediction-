# Machine-learning-Project-survival-rate-prediction-

So in this project , we picked up the databse from kaggle named titanic daabse 
	1.	Loading the dataset:
We loaded the dataset using pandas:
import pandas as pd
df = pd.read_csv('titanic.csv')
	2.	Checking missing values:
To check how many values were missing in each column:
df.isnull().sum()
	3.	Handling missing numerical values:
We filled missing values in the ‘Age’ and ‘Fare’ columns using the median:
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
	4.	Handling missing categorical values:
We filled missing values in the ‘Embarked’ column using the mode:
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
	5.	Encoding categorical variables:
We encoded the ‘Sex’ column using label encoding:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

For one-hot encoding (e.g. for ‘Embarked’):
df = pd.get_dummies(df, columns=['Embarked'])
	6.	Normalizing continuous features:
We normalized ‘Fare’ and ‘Age’ using MinMaxScaler:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
	7.	Visualizing the data:
We used seaborn and matplotlib for plotting graphs:
import seaborn as sns
import matplotlib.pyplot as plt

