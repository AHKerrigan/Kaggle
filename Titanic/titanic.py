import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class TitanicModel():
	def __init__(self, num_features):
		# Placeholders for the input data
		self.X = tf.placeholder(tf.float32)
		self.Y = tf.placeholder(tf.float32)

		# Define the weights, biases, as well as the prediction functoin
		self.W = tf.Variable(tf.zeros([num_features, 1]))
		self.b = tf.Variable(tf.zeros([1]))
		self.z = tf.matmul(self.X, self.W) + self.b

		

		# Define the loss function as the logistic crossentropy function
		self.loss_function = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z, labels=self.Y))

		# Define the training self as using gradient descent as the loss function 
		self.train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss_function)

		# Functions for rounding predictions as well as evaluating which predictions are correct
		self.prediction = tf.round(tf.sigmoid(self.z))
		self.correct = tf.cast(tf.equal(self.prediction, self.Y), dtype=tf.float32)

		self.accuracy = tf.reduce_mean(self.correct)

		self.sess = tf.Session()

	
	def train(self, X_train, y_train):

		self.sess.run(tf.global_variables_initializer())
		for epoch in range(3000):

			self.sess.run(self.train_step, feed_dict={self.X: X_train, self.Y: y_train})
			temp_loss = self.sess.run(self.loss_function, feed_dict={self.X: X_train, self.Y: y_train})

			if epoch % 200 == 0:
				print("Loss at epoch", epoch, ": ", str(temp_loss))
		print("Completed")
		print(self.sess.run(tf.sigmoid(self.z), feed_dict={self.X: X_train}))

	
	def evaluate(self, X_eval, y_eval):
		evaluation_accuracy = self.sess.run(self.accuracy, feed_dict={self.X: X_eval, self.Y: y_eval})
		print(evaluation_accuracy)
	
	def predict(self, X_test): 
		return self.sess.run(self.prediction, feed_dict={self.X: X_test})




def load_data(filename):
	columns = pd.Index(["Pclass", "Sex", "SibSp", "Parch", "Survived"], name='cols')

	data = pd.DataFrame(pd.read_csv(filename), columns = columns)
	data = data.replace("female", 1)
	data = data.replace("male", 0)

	X = data.drop(["Survived"], axis=1)
	X = X.dropna()
	Y = data["Survived"]
	Y = Y.dropna()

	return  X, Y

if __name__ == "__main__":
	trainX, trainY = load_data("data/train.csv")
	testX, _ = load_data("data/test.csv")
 
	trainX, valX = trainX[:600], trainX[600:]
	trainY, valY = trainY[:600], trainY[600:]

	# We grab the number of features so we can more easily edit what features we use 
	num_features = len(trainX.columns)
 
	new_model = TitanicModel(num_features)
	new_model.train(trainX, trainY)
	#new_model.evaluate(valX, valY)

	#test = new_model.predict(testX)

	new_model.sess.close()

