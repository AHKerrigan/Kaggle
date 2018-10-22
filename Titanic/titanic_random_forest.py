import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import shuffle

def load_training_data(filename):
	columns = pd.Index(["Name","Title", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"], name='cols')

	data = pd.DataFrame(pd.read_csv(filename), columns = columns)
	print(data.size)
	
	# Create the title feature 
	data["Title"] = data.Name.str.extract(' ([A-Za-z]+)\.')

	data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

	data['Title'] = data['Title'].replace('Mlle', 'Miss')
	data['Title'] = data['Title'].replace('Ms', 'Miss')
	data['Title'] = data['Title'].replace('Mme', 'Mrs')

	# Map the remaining title feature to values 
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
	data['Title'] = data['Title'].map(title_mapping)
	data['Title'] = data['Title'].fillna(0)

	data = data.drop("Name", axis=1)

	# Replace female with binary
	data = data.replace("female", 1)
	data = data.replace("male", 0)

	# Perform similar replacement on Embark
	data = data.replace("Q", 0)
	data = data.replace("S", 1)
	data = data.replace("C", 2)

	# Age has a lot of NaNs, so we'll replace them with the average
	# So first, we calculate the average
	total = 0
	m = 0

	for index, rows in data.iterrows():
		if (rows.isnull().values.any()) == False:
			total += rows["Age"]
			m += 1
	
	# Fill the age with the average
	# Also fill some stray embarked Na
	data = data.fillna(value={"Age":(total/m), "Embarked": 0})

	shuffle(data)

	trainY = data["Survived"]
	trainX = data.drop("Survived", axis=1)
	return trainX, trainY

	"""
	Version where we cross validated the data
	live = data[data["Survived"] == 1]
	dead = data[data["Survived"] == 0]

	split = 200
	trainY = [live["Survived"][:split], dead["Survived"][:split]]
	trainY = pd.concat(trainY)
	trainX = live.drop("Survived", axis=1)[:split], dead.drop("Survived", axis=1)[:split]
	trainX = pd.concat(trainX)

	valY = [live["Survived"][split:], dead["Survived"][split:]]
	valY = pd.concat(valY)
	valX = live.drop("Survived", axis=1)[split:], dead.drop("Survived", axis=1)[split:]
	valX = pd.concat(valX)
	"""

def load_testing_data(filename):
	columns = pd.Index(["Name", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"], name='cols')

	data = pd.DataFrame(pd.read_csv(filename), columns = columns)

	# Create the title feature 
	data["Title"] = data.Name.str.extract(' ([A-Za-z]+)\.')

	data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

	data['Title'] = data['Title'].replace('Mlle', 'Miss')
	data['Title'] = data['Title'].replace('Ms', 'Miss')
	data['Title'] = data['Title'].replace('Mme', 'Mrs')

	# Map the remaining title feature to values 
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
	data['Title'] = data['Title'].map(title_mapping)
	data['Title'] = data['Title'].fillna(0)

	data = data.drop("Name", axis=1)
	
	# Replace female with binary
	data = data.replace("female", 1)
	data = data.replace("male", 0)

	# Perform similar replacement on Embark
	data = data.replace("Q", 0)
	data = data.replace("S", 1)
	data = data.replace("C", 2)

	print(data)
	data = data.fillna(value={"Age":(30), "Embarked": 0, "Fare": 10})

	return data

def output_results(results):
	index = []
	for i in range(418):
		index.append(i + 892)
	
	results = pd.DataFrame(results, index=index)
	results.to_csv("results.csv", sep='\t')		


if __name__ == "__main__":
	
	
	model = RandomForestClassifier(n_estimators=1000, n_jobs=1)

	trainX, trainY = load_training_data("data/train.csv")
	

	model.fit(trainX, trainY)
	
	testX = load_testing_data("data/test.csv")
	
	results = model.predict(testX)
	output_results(results)

	
	
	
