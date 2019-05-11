import csv

import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from keras.layers import Dense, Input, Dropout, Activation, Flatten, Softmax
from keras.models import Sequential, Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, backend
from keras.callbacks import EarlyStopping, ModelCheckpoint

from xgboost import XGBClassifier

import tensorflow as tf

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
	sol_train = train[['EventId', 'Label', 'Weight']]
	X_test = pd.read_csv(test_file, usecols=range(1,31)).to_numpy()
	rand = pd.read_csv(rand_file)
	return X_train, Y_train, w_train, sol_train, X_test, rand


def split_data(X, Y, w, sol, test_size=0.1, seed=420):
	X_train, X_val, Y_train, Y_val, _, w_val, _, sol_test = (
		train_test_split(X, Y, w, sol, test_size=test_size, random_state=seed))
	return X_train, X_val, Y_train, Y_val, w_val, sol_test



def create_submission(preds, rand):
	"""Creates submissions given classifier's predictions.

	Parameters
	----------
	preds : array-like, shape of (N)
		Predictions for each test example, between 0-1.

	rand : Pandas Dataframe, shape of (N, 3)
		Random submission, Header (EventId, RankOrder, Class).

	Returns
	-------
	sub : Pandas Dataframe, shape of (N, 3)
		Edited submission, Header (EventId, RankOrder, Class).
	"""
	rank = np.argsort(preds) + 1
	print(rank)
	binary_preds = pd.DataFrame((preds > 0.5))
	binary_preds = binary_preds.replace([True, False], ['s', 'b'])
	print(binary_preds)
	rand['RankOrder'] = rank
	rand['Class'] = binary_preds
	return rand[['EventId', 'RankOrder', 'Class']]


def evaluate(submission, solution):
	"""Evaluates submission given solution.

	Parameters
	----------
	submission : Pandas Dataframe, shape of (N, 3)
		Header (EventId, RankOrder, Label).

	solution : Pandas Dataframe, shape of (N, 3)
		Same header as submission.

	Returns
	-------
	ams : Float
		AMS score.
	"""
	return AMS_metric(submission, solution)	


def build_NN(input_shape=30, h_size=600):
	"""Builds neural network model.

	Parameters
	----------
	input_shape : Positive int, default=30
		Number of features in one example.

	h_size : Positive int, default=600
		Size of hidden layers.

	Returns
	-------
	model : Keras Sequential object
		Untrained model object.
	"""
	model = Sequential()
	layers = [Dense(h_size, activation='relu', input_shape=(input_shape,),
					kernel_regularizer=regularizers.l1_l2(l1=5e-6, l2=5e-5)), 
			  Dropout(0.5),
			  Dense(h_size, activation='relu'),
			  Dropout(0.5),
			  Dense(h_size, activation='relu'),
			  Dense(1, activation='sigmoid')]
	for layer in layers:
		model.add(layer)
	print(model.summary())
	return model


def train_NN(model, X_train, Y_train, X_val, Y_val, w_val, filename):
	"""Builds neural network model.

	Parameters
	----------
	model : Keras Sequential object
		Untrained model object.

	X_train : Array-like of floats, shape of (N, F)
		Training examples.

	Y_train : Array-like of ints, shape of (N)
		Training labels.

	X_val : Array-like of floats, shape of (M, F)
		Validation examples.

	Y_val : Array-like of ints, shape of (M)
		Validation labels.

	w_val : Array-like of floats, shape of (M)
		Weights for each example, used in computing loss function.

	filename : Str
		File to save trained model, ideally ending in ".h5"

	Returns
	-------
	model : Keras Sequential object
		Untrained model object.
	"""
	model.compile(loss='binary_crossentropy', optimizer='adam',
				  metrics=['accuracy'])
	es = EarlyStopping(monitor='val_loss',
	                   min_delta=0,
	                   patience=0,
	                   verbose=0, mode='auto')
	checkpoint = ModelCheckpoint(filename, monitor='val_loss', 
								 verbose=0, save_best_only=True, mode='auto')
	model.fit(X_train, Y_train, batch_size=64, epochs=1,
			  callbacks=[es,checkpoint], validation_data=(X_val, Y_val, w_val))
	final_model = load_model(filename)
	return final_model


def predict_NN(model, X_test):
	return model.predict(X_test)


def build_xgb(max_depth=8, n_trees=150, eta=0.01):
	model = XGBClassifier(max_depth=max_depth, learning_rate=eta, n_estimators=150)
	return model


def train_xgb(model, X_train, Y_train, X_val, Y_val, w_val, filename):
	model.fit(
		X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=1,
		sample_weight_eval_set=[w_val], early_stopping_rounds=10, eval_metric='auc')
	joblib.dump(model, filename)
	loaded_model = joblib.load(filename)
	return loaded_model


def predict_xgb(model, X_test):
	return model.predict(X_test, ntree_limit=model.best_ntree_limit)



if __name__ == '__main__':
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	X, Y, w, sol, X_test, rand = load_data(TRAIN_FILE, 
										   TEST_FILE, 
										   RAND_FILE)
	X_train, X_val, Y_train, Y_val, w_val, sol_test = split_data(X, Y, w, sol)
	model = build_NN()
	trained = train_NN(model, X_train, Y_train, X_val, Y_val, w_val, '../models/modelNN_2.h5')
	preds = predict_NN(trained, X_val)
	sub = create_submission(preds, sol_test)
	print(sub)
	# print(evaluate(sub, sol_test))

	# model = build_NN()
	# trained = train_NN(model, X_train, Y_train, X_val, Y_val, w_val, 'model1.h5')
	# preds = trained.predict(X_test)
	# sub = create_submission(preds, rand)
	# print(evaluate(sub, sol_test))
	
	# rand_preds = np.random.rand(rand.shape[0])
	# create_submission(rand_preds, rand)