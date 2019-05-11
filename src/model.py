import csv, os

import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt 

from sklearn.model_selection import KFold
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
    X_train = train.iloc[:, 1:31]
    Y_train = (train.iloc[:, 32] == 's').to_numpy(dtype=int)
    w_train = train.iloc[:, 31].to_numpy()
    sol_train = train[['EventId', 'Label', 'Weight']]
    X_test = pd.read_csv(test_file, usecols=range(1,31))
    rand = pd.read_csv(rand_file)
    return X_train, Y_train, w_train, sol_train, X_test, rand


def split_data(train_indices, val_indices, X, Y, w, sol):
    """Splits data into training and validation sets.

    Parameters
    ----------
    train_indices : Array-like of ints
        Indices for training set.

    val_indices : Array-like of ints
        Indices for validation set.

    X : Pandas Dataframe, shape of (N, F)
        Full training data.

    Y : Numpy array, shape of (N)
        Full training labels (binary).

    w : Numpy array, shape of (N)
        Weights for each training example.

    sol : Pandas Dataframe, shape of (N, 3)
        Solution for training data, used for local validation/scoring.

    Returns
    -------
    X_train : Pandas Dataframe, shape of (J, F)
        Training data.

    X_val : Pandas Dataframe, shape of (K, F)
        Validation data.

    Y_train : Numpy array, shape of (J)
        Training labels (binary).

    Y_val : Numpy array, shape of (K)
        Validation labels (binary).

    w_val : Numpy array, shape of (K)
        Validation example weights.

    sol : Pandas Dataframe, shape of (K, 3)
        Solution for validation data, used for local validation/scoring. 
    """
    return (X.iloc[train_indices], X.iloc[val_indices], Y[train_indices], 
            Y[val_indices],   w[val_indices], sol.iloc[val_indices])


def add_features(*dfs):
    """Adds hand-picked features to input dataframes.

    Parameters
    ----------
    dfs : Iterable, of Pandas Dataframes
        Input data to augment. 

    Returns
    -------
    augmented_dfs : Tuple, of Pandas Dataframes
        Augmented data, adds 9 new columns.
        Five mass features and four radian features: 
        - MASS_[1-5]
        - 'RAD_min_tltm', 'RAD_min_tltmlm', 'RAD_min_tmlm', 'RAD_min_lm'
    """
    
    def add_mass_features(df):
        """Adds mass-based features to a single Pandas Dataframe"""
        tau = calc_p(df['PRI_tau_pt'], df['PRI_tau_phi'], df['PRI_tau_eta'])
        lep = calc_p(df['PRI_lep_pt'], df['PRI_lep_phi'], df['PRI_lep_eta'])
        jet_1 = calc_p(df['PRI_jet_leading_pt'], 
                       df['PRI_jet_leading_phi'], 
                       df['PRI_jet_leading_eta'])
        jet_2 = calc_p(df['PRI_jet_subleading_pt'], 
                       df['PRI_jet_subleading_phi'], 
                       df['PRI_jet_subleading_eta'])
        df['MASS_1'] = np.log(1 + m_inv(tau, jet_1))
        df['MASS_2'] = np.log(1 + m_inv(tau, jet_2))
        df['MASS_3'] = np.log(1 + m_inv(tau, lep))
        df['MASS_4'] = np.log(1 + m_tr(tau, jet_1))
        df['MASS_5'] = np.log(1 + m_tr(tau, jet_2))
        return df
    
    def add_radian_features(df):
        """Adds radian-based features to a single Pandas Dataframe"""
        diff_tau_lep = (
            ((df.PRI_tau_phi - df.PRI_lep_phi) - np.pi) % (2*np.pi)) - np.pi

        diff_tau_met = (
            ((df.PRI_tau_phi - df.PRI_met_phi) - np.pi) % (2*np.pi)) - np.pi

        diff_lep_met = (
            ((df.PRI_lep_phi - df.PRI_met_phi) - np.pi) % (2*np.pi)) - np.pi

        df['RAD_min_tltm']   = np.minimum(diff_tau_lep, diff_tau_met)
        df['RAD_min_tltmlm'] = np.minimum(df.RAD_min_tltm, diff_lep_met)
        df['RAD_min_tmlm']   = np.minimum(diff_tau_met, diff_lep_met)
        df['RAD_min_lm']     = diff_lep_met
        return df
    
    return (add_mass_features(add_radian_features(df)) for df in dfs)


def calc_p(pt, phi, eta):
    """Calculates momentum of a particle.

    Parameters
    ----------
    pt : Pandas series of floats
        Transverse momentum.

    phi : Pandas series of floats
        Azimuthal angle.

    eta : Pandas series of floats
        Pseudorapidity.

    Returns
    -------
    df_p : Pandas Dataframe
        Momentum in each direction, Header (p_x, p_y, p_z)
    """
    p_x = pt * np.cos(phi)
    p_y = pt * np.sin(phi)
    p_z = pt * np.sinh(eta)
    return pd.DataFrame({'p_x': p_x, 'p_y': p_y, 'p_z': p_z})


def m_inv(a, b):
    """Calculates invariant mass of two particles a and b.

    Parameters
    ----------
    a : Pandas Dataframe
        Momentum in each direction, Header (p_x, p_y, p_z)

    b : Pandas Dataframe
        Momentum in each direction, Header (p_x, p_y, p_z)

    Returns
    -------
    m : Pandas Series
        Invariant mass of the two particles.
    """

    a_x, a_y, a_z = a['p_x'], a['p_y'], a['p_z']
    b_x, b_y, b_z = b['p_x'], b['p_y'], b['p_z']
    norm_a = np.sqrt(a_x**2 + a_y**2 + a_z**2)
    norm_b = np.sqrt(b_x**2 + b_y**2 + b_z**2)
    sq_sum = (a_x + b_x)**2 + (a_y + b_y)**2 + (a_z + b_z)**2
    return np.sqrt((norm_a + norm_b)**2 - sq_sum)


def m_tr(a, b):
    """Calculates transverse mass of two particles a and b.

    Parameters
    ----------
    a : Pandas Dataframe
        Momentum in each direction, Header (p_x, p_y, p_z)

    b : Pandas Dataframe
        Momentum in each direction, Header (p_x, p_y, p_z)

    Returns
    -------
    m : Pandas Series
        Transverse mass of the two particles.
    """
    a_x, a_y = a['p_x'], a['p_y']
    b_x, b_y = b['p_x'], b['p_y']
    norm_a = np.sqrt(a_x**2 + a_y**2)
    norm_b = np.sqrt(b_x**2 + b_y**2)
    sq_sum = (a_x + b_x)**2 + (a_y + b_y)**2
    return np.sqrt((norm_a + norm_b)**2 - sq_sum)    


def next_model_filename(model_type):
    """Returns next model filename to save to.

    Parameters
    ----------
    model_type : Str, either 'NN' or 'XGB'

    Returns
    -------
    filename : Str
        The next model filename.
    """
    file_nums = [int(file.replace('model_', '').replace('.h5', ''))
                     for file in os.listdir("../models/" + model_type)]
    if not file_nums: 
        return '../models/' + model_type + '/model_0.h5'
    filename = ('../models/' + model_type + '/model_' + 
                str(max(file_nums) + 1) + '.h5')
    return filename


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
    rand['RankOrder'] = rank
    rand['Label'] = (preds > 0.5)
    rand['Label'] = rand['Label'].replace([True, False], ['s', 'b'])
    return rand[['EventId', 'RankOrder', 'Label']]


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
        Trained model object.
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
    """Returns predictions of NN model on X_test as a Numpy array."""
    return model.predict(X_test).flatten()


def predict_n_NN(X_test):
    """
    Returns predictions on X_test using an ensemble of 
    neural net models saved on disk.
    """
    preds = np.zeros(X_test.shape[0])
    num_models = 0
    for model_filename in os.listdir('../models/NN'):
        model = load_model(model_filename)
        preds += predict_NN(model, X_test)
        num_models += 1
    return preds / num_models


def build_xgb(max_depth=8, n_trees=300, eta=0.01):
    """Builds XGB model.

    Parameters
    ----------
    max_depth : Int, default=8
        Maximum depth of each decision tree stump.

    n_trees : Int, default=300
        Maximum number of tree estimators.

    eta : Float, default=0.01
        Learning rate.

    Returns
    -------
    model : XGBClassifier object
        Untrained XGB model.
    """
    return XGBClassifier(
            max_depth=max_depth, learning_rate=eta, n_estimators=n_trees)
    

def train_xgb(model, X_train, Y_train, X_val, Y_val, w_val, filename):
    """Trains an XGB classifier and saves it to filename.

    Parameters
    ----------
    model : XGBClassifier object
        Untrained XGB model.

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
    model : XGBClassifier object
        Trained model object.
    """
    model.fit(
        X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=True,
        sample_weight_eval_set=[w_val], early_stopping_rounds=10, eval_metric='auc')
    joblib.dump(model, filename)
    loaded_model = joblib.load(filename)
    return loaded_model


def predict_xgb(model, X_test):
    """Returns predictions of XGB model on X_test as a Numpy array."""
    return model.predict(X_test, ntree_limit=model.best_ntree_limit)


def predict_n_xgb(X_test):
    """
    Returns predictions on X_test using an ensemble of 
    gradient-boosted tree models saved on disk.
    """
    preds = np.zeros(X_test.shape[0])
    num_models = 0
    for model_filename in os.listdir('../models/XGB'):
        model = joblib.load(model_filename)
        preds += predict_xgb(model, X_test)
        num_models += 1
    return preds / num_models


def train_n_models(N, build, train, model_type, X, Y, w, sol):
    """Trains N models of a certain type and saves them to disk.

    Parameters
    ----------
    N : Int, > 0
        Number of models to train.

    build : Function to build the model, either build_NN or build_xgb.

    train : Function to train the model, either train_NN or train_xgb.

    model_type : Str, either 'NN' or 'XGB'

    X : Pandas Dataframe, shape of (N, F)
        Full training data.

    Y : Numpy array, shape of (N)
        Full training labels (binary).

    w : Numpy array, shape of (N)
        Weights for each training example.

    sol : Pandas Dataframe, shape of (N, 3)
        Solution for training data, used for local validation/scoring.
    
    Returns: None
    """ 
    for train_ind, val_ind in KFold(n_splits=N).split(Y):
        X_train, X_val, Y_train, Y_val, w_val, sol_test = (
                split_data(train_ind, val_ind, X, Y, w, sol))
        model = build()
        filename = next_model_filename(model_type)
        # Saves trained model to disk
        _ = train(model, X_train, Y_train, X_val, Y_val, w_val, filename)


if __name__ == '__main__':
    X, Y, w, sol, X_test, rand = load_data(TRAIN_FILE, 
                                           TEST_FILE, 
                                           RAND_FILE)
    X, X_test = add_features(X, X_test)
    # N = 20
    # train_n_models(N, build_xgb, train_xgb, 'XGB', X, Y, w, sol)
    preds = predict_n_xgb(X_test)
    sub = create_submission(preds, rand)
    sub.to_csv('xgb_submission_1.csv')
    # print(evaluate(sol_test, sub))