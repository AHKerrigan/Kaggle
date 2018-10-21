import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import shuffle

def load_training_data(filename):
	columns = pd.Index(["Pclass", "Sex", "SibSp", "Parch", "Fare", "Survived"], name='cols')

	data = pd.DataFrame(pd.read_csv(filename), columns = columns)
	data = data.replace("female", 1)
	data = data.replace("male", 0)
	data = shuffle(data)

	live = data[data["Survived"] == 1].dropna()
	dead = data[data["Survived"] == 0].dropna()

	trainY = [live["Survived"][:300], dead["Survived"][:300]]
	trainY = pd.concat(trainY)
	trainX = live.drop("Survived", axis=1)[:300], dead.drop("Survived", axis=1)[:300]
	trainX = pd.concat(trainX)

	valY = [live["Survived"][300:], dead["Survived"][300:]]
	valY = pd.concat(valY)
	valX = live.drop("Survived", axis=1)[300:], dead.drop("Survived", axis=1)[300:]
	valX = pd.concat(valX)

	return trainX, trainY, valX, valY

def load_testing_data(filename):
	columns = pd.Index(["Pclass", "Sex", "SibSp", "Parch", "Fare"], name='cols')

	data = pd.DataFrame(pd.read_csv(filename), columns = columns)
	data = data.replace("female", 1)
	data = data.replace("male", 0)
	data = data.fillna(35)

	return data

def output_results(results):
	index = []
	for i in range(418):
		index.append(i + 892)
	
	results = pd.DataFrame(results, index=index)
	results.to_csv("results.csv", sep='\t')


if __name__ == "__main__":
	
	
	model = RandomForestClassifier(n_estimators=20, n_jobs=1)

	trainX, trainY, valX, valY = load_training_data("data/train.csv")

	model.fit(trainX, trainY)
	model_score = model.score(valX, valY)
	
	testX = load_testing_data("data/test.csv")
	results = model.predict(testX)
	print(model_score)
	# I want at least 80% on the testing stage before outputting
	if model_score >= 0.8:
		output_results(results)