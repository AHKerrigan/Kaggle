import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical


# Sample rate for the data is 22050

def wav2mfcc(file_path, max_pad_len=775):
	try:
		wave, sr = librosa.load(file_path, mono=True)
	except FileNotFoundError:
		print(file_path, "not found, moving on....")
		return np.asarray([])
	mfcc = librosa.feature.mfcc(wave, sr=22050)
	pad_width = max_pad_len - mfcc.shape[1]
	mfcc = np.pad(mfcc, pad_width=((0,0), (0, pad_width)), mode='constant')
	return mfcc

def load_data():
	mfcc_vectors = []
	labels = []
	for i in range(1, 167):
		filename = "data/cat_" + str(i) + ".wav"
		temp = wav2mfcc(filename)
		if temp.any():
			mfcc_vectors.append(temp)
			labels.append(0)
	for i in range(0, 112):
		filename = "data/dog_barking_" + str(i) + ".wav"
		temp = wav2mfcc(filename)
		if temp.any():
			mfcc_vectors.append(temp)
			labels.append(1)
	
	

	return train_test_split(np.asarray(mfcc_vectors), np.asarray(labels), shuffle=True, test_size=0.2)

if __name__ == "__main__":

	train_X, test_X, train_y, test_y = load_data()
	train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
	test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)

	print(np.shape(train_X))
	
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 775, 1)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
				optimizer=tensorflow.keras.optimizers.Adadelta(),
				metrics=['accuracy'])	
	
	model.fit(train_X, train_y, epochs=300, validation_data=(test_X, test_y))
