
# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# load data
# dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")



# split data into X and y
# X = dataset[:,0:8]
# Y = dataset[:,8]


######################################### vvv GARBAGE
n = 100
X = []
Y = []
for i in range (0, n):
#    X.append(np.array(range(n, n+8)))
    X.append(np.array(range(i, i+8)))
    Y.append(i%3)

X = np.array(X)
Y = np.array(Y)
######################################### ^^^ GARBAGE


print("X :")
print(X)

print("\nY: ")
print(Y)


# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


def build_xgb(max_depth=8, n_trees=150, eta=0.01)
	model = XGBClassifier(max_depth=max_depth, learning_rate=eta, n_estimators=150)
	return model


def train_xgb(model, X_train, Y_train, X_val, Y_val, w_val, filename):
	model.fit(
		X_train, Y_train, eval_set=[(X_val, Y_val)], 
		sample_weight_eval_set=[w_val], early_stopping_rounds=10, eval_metric='auc')
	joblib.dump(model, filename)
	loaded_model = joblib.load(filename)
	return loaded_model


def predict_xgb(model, X_test):
	return model.predict(X_test, ntree_limit=model.best_ntree_limit)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

######
print("\ny_pred: ")
print(y_pred)

print("\npredictions: ")
print(predictions)

######

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

