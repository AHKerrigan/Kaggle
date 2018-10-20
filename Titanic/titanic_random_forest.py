import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def load_data(filename):
	columns = pd.Index(["Pclass", "Sex", "SibSp", "Parch", "Fare", "Survived"], name='cols')

	data = pd.DataFrame(pd.read_csv(filename), columns = columns)
	data = data.replace("female", 1)
	data = data.replace("male", 0)

	live = pd.DataFrame(data.Survived == 1)
	dead = pd.DataFrame(data.Survived == 0)

	
	print(live)


if __name__ == "__main__":
	
	"""
	model = RandomForestRegressor(n_jobs=1)

	X, Y = load_data("data/train.csv")

	X_train = X[:500]
	Y_train = Y[:500]
	X_val = X[500:]
	Y_val = Y[500:]

	print(X_train)
	print(Y_train)

	#model.fit(X_val, Y_val)
	#print(model.score(X_val, Y_val))
	"""
	load_data("data/train.csv")