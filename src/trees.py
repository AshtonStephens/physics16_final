
# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

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

