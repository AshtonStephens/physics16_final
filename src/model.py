import csv

import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Input, Dropout, Activation, Flatten, Softmax
from keras.models import Sequential, Model
from keras import initializers, regularizers, constraints, optimizers, layers, backend
from keras.callbacks import EarlyStopping, ModelCheckpoint



from HiggsBosonCompetition_AMSMetric_rev1 import AMS_metric


TRAIN_FILE = '../data/training.csv'
TEST_FILE = '../data/test.csv'
RAND_FILE = '../data/random_submission.csv'


def load_data(train_file, test_file, rand_file):
	"""Loads in data.

	Parameters
	----------
	train_file: string 
		Path to training data file
	
	test_file: string 
		Path to testing data file
	
	rand_file: string 
		Path to random submission file

	Returns
	-------
	X_train: Numpy array, shape (N, F)
		Training features

	Y_train: Numpy array, shape (N)
		Training labels, binary (0 = background, 1 = signal)

	w_train: Numpy array, shape (N)
		Weights for each example for loss function

	X_test: Numpy array, shape (M, F)
		Testing features

	rand : Pandas dataframe, shape (M, 3)
		Submission template, columns are event_id, rank_order, label
	"""
	train = pd.read_csv(train_file)
	X_train = train.iloc[:, 1:31].to_numpy()
	Y_train = (train.iloc[:, 32] == 's').to_numpy(dtype=int)
	w_train = train.iloc[:, 31].to_numpy()
	sol_train = train[['EventId', 'Weight', 'Label']]
	X_test = pd.read_csv(test_file, usecols=range(1,31)).to_numpy()
	rand = pd.read_csv(rand_file)
	return X_train, Y_train, w_train, sol_train, X_test, rand


def build_NN():
	model = Sequential()
	layers = [Dense(600, activation='relu', input_shape=(30,),
					kernel_regularizer=regularizers.l1_l2(l1=5e-6, l2=5e-5)), 
			  Dropout(0.5),
			  Dense(600, activation='relu'),
			  Dropout(0.5),
			  Dense(600, activation='relu'),
			  Dense(1, activation='sigmoid')]
	for layer in layers:
		model.add(layer)
	return model


def train_NN(model, X_train, Y_train, X_val, Y_val, w_val, filename):
	model.compile(loss='binary_crossentropy', optimizer='adam',
				  metrics=['accuracy'])
	es = EarlyStopping(monitor='val_loss',
	                   min_delta=0,
	                   patience=3,
	                   verbose=0, mode='auto')
	checkpoint = ModelCheckpoint(filename, monitor='val_loss', 
								 verbose=0, save_best_only=True, mode='auto')
	model.fit(X_train, Y_train, batch_size=64, epochs=30,
			  callbacks=[es,checkpoint], validation_data=(X_val, Y_val, w_val))
	final_model = load_model(filename)
	return final_model


def create_submission(preds, rand):
	rank = np.argsort(preds) + 1
	binary_preds = pd.DataFrame((preds > 0.5), dtype='str')
	binary_preds = binary_preds.replace(['True', 'False'], ['s', 'b'])
	rand['RankOrder'] = rank
	rand['Class'] = binary_preds
	return rand


def evaluate(submission, solution):
	return AMS_metric(submission, solution)	


if __name__ == '__main__':
	X, Y, w, sol, X_test, rand = load_data(TRAIN_FILE, 
												 TEST_FILE, 
												 RAND_FILE)
	X_train, X_val, Y_train, Y_val, _, w_val, _, sol_test = train_test_split(X, Y, w, sol,
															test_size=0.1, 
															random_state=420)
	model = build_NN()
	trained = train_NN(model, X_train, Y_train, X_val, Y_val, w_val, 'model1.h5')
	preds = trained.predict(X_test)
	sub = create_submission(preds, rand)
	print(evaluate(sub, sol_test))
	
	# rand_preds = np.random.rand(rand.shape[0])
	# create_submission(rand_preds, rand)