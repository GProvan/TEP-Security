'''
https://github.com/Minqi824/ADBench/blob/main/datasets/Classical/31_satimage-2.npz
https://github.com/Minqi824/ADBench/blob/main/datasets/Classical/38_thyroid.npz
https://github.com/Minqi824/ADBench/blob/main/datasets/Classical/5_campaign.npz
https://github.com/Minqi824/ADBench/blob/main/datasets/Classical/10_cover.npz
https://github.com/Minqi824/ADBench/blob/main/datasets/Classical/23_mammography.npz
https://github.com/Minqi824/ADBench/blob/main/datasets/Classical/32_shuttle.npz
https://github.com/Minqi824/ADBench/blob/main/datasets/Classical/2_annthyroid.npz
https://github.com/Minqi824/ADBench/blob/main/datasets/Classical/13_fraud.npz
'''
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import logging
import abc
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
import tqdm

from torch.autograd import Variable


import sklearn
import itertools
import operator
import warnings
from collections import Counter
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score

import ruptures as rpt
import hyperopt as hp
from pyod.models.ecod import ECOD
from pyod.models.abod import ABOD
from pyod.models.lunar import LUNAR
from pyod.models.vae import VAE
from pyod.models.alad import ALAD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.knn import KNN
from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, hp,Trials
from pyod.models.deep_svdd import DeepSVDD

from preprocessADBench import preprocess_datasets

sys.path.append('../')
from cadlae.preprocessComparison import *
X_train, y_train, X_test, y_test, col_names = preprocess_datasets("adbench","/content/23_mammography.npz",0.15 )


def _precision(tp, fp):
	'''
	Precision = TP / (TP + FP)
	:param tp: True Positive
	:param fp: False Positive
	:return: precision

	Usage of the function:
	-> _precision(10, 5)
	'''
	pre = tp / float(tp + fp)
	return pre


def _recall(tp, fn):
	'''
	Recall = TP / (TP + FN)
	:param tp: True Positive
	:param fn: False Negative
	:return: recall

	Usage of the function:
	-> _recall(10, 5)
	'''
	rec = tp / float(tp + fn)
	return rec


def _f1(pre, rec):
	'''
	F1 = 2 * (Precision * Recall) / (Precision + Recall)
	:param pre: Precision
	:param rec: Recall
	:return: F1 Score

	Usage of the function:
	-> _f1(0.5, 0.5)
	'''
	f1 = 2 * (pre * rec) / (pre + rec)
	return f1


def print_results(accuracy, pre, rec, f1, roc):
	'''
	Print the results of the model
	:param accuracy: Accuracy of the model
	:param pre: Precision of the model
	:param rec: Recall of the model
	:param f1: F1 Score of the model
	:param roc: ROC AUC Score of the model
	:return: None

	Usage of the function:
	-> print_results(0.5, 0.5, 0.5, 0.5, 0.5)
	'''
	print("Accuracy: {}%".format(round(accuracy * 100, 2)))
	print("Precision: {}%".format(round(pre * 100, 2)))
	print("Recall: {}%".format(round(rec * 100, 2)))
	print("F1 Score: {}%".format(round(f1 * 100, 2)))
	print("ROCAUC: {}%".format(round(roc * 100, 2)))


def calculate_results_print(y_test, y_pred):
	'''
	Calculate the results of the model and print them
	:param y_test: Test set
	:param y_pred: Predicted set
	:return: None

	Usage of the function:
	-> calculate_results_print(y_test, y_pred)
	'''
	cm = confusion_matrix(y_test, y_pred)
	TP = cm[0, 0]
	TN = cm[1, 1]
	FP = cm[0, 1]
	FN = cm[1, 0]
	accuracy = accuracy_score(y_test, y_pred)
	precision = _precision(TP, FP)
	recall = _recall(TP, FN)
	f1 = _f1(precision, recall)
	roc = roc_auc_score(y_test, y_pred)
	print_results(accuracy, precision, recall, f1, roc)


# Validated Spaces
knn_space = {"contamination": hp.uniform("contamination", 0.05, 0.25),
			 "n_neighbors": hp.choice('n_neighbors', np.arange(2, 21, dtype=int)),
			 "leaf_size": hp.quniform("leaf_size", 15, 45, 1),
			 "radius": hp.uniform("radius", 0.5, 1.5),
			 "method": hp.choice("method", ['largest', 'mean', 'median']),
			 "algorithm": hp.choice("algorithm", ['auto', 'ball_tree', 'kd_tree'])}

vae_space = {"gamma": hp.uniform("gamma", 0.5, 3),
			 "validation_size": hp.uniform("validation_size", 0.05, 0.2),
			 "capacity": hp.uniform("capacity", 0.0, 0.2),
			 "batch_size": hp.choice('batch_size', np.arange(32, 256, dtype=int)),
			 "epochs": hp.choice('epochs', np.arange(100, 300, dtype=int)),
			 "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.4),
			 "contamination": hp.uniform("contamination", 0.05, 0.25),
			 "l2_regularizer": hp.uniform("l2_regularizer", 0.1, 0.7)}

deep_svdd_space = {"validation_size": hp.uniform("validation_size", 0.05, 0.2),
				   "batch_size": hp.choice('batch_size', np.arange(32, 256, dtype=int)),
				   # "epochs" : hp.choice('epochs', np.arange(100, 250, dtype=int)),
				   "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.4),
				   "contamination": hp.uniform("contamination", 0.05, 0.25),
				   "l2_regularizer": hp.uniform("l2_regularizer", 0.1, 0.7)}

alad_space = {"contamination": hp.uniform("contamination", 0.05, 0.25),
			  "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.7),
			  "batch_size": hp.choice('batch_size', np.arange(32, 256, dtype=int)),
			  "epochs": hp.choice('epochs', np.arange(500, 800, dtype=int)),
			  "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.7),
			  "learning_rate_gen": hp.uniform("learning_rate_gen", 0.0001, 0.01),
			  "learning_rate_disc": hp.uniform("learning_rate_disc", 0.0001, 0.01)}

ecod_space = {"contamination": hp.uniform("contamination", 0.05, 0.25)}

abod_space = {"contamination": hp.uniform("contamination", 0.05, 0.25),
			  "n_neighbors": hp.choice('n_neighbors', np.arange(5, 30, dtype=int))}

lunar_space = {"model_type": hp.choice("model_type", ['WEIGHT', 'SCORE']),
			   "n_neighbors": hp.choice('n_neighbors', np.arange(3, 15, dtype=int)),
			   "negative_sampling": hp.choice("negative_sampling", ['UNIFORM', 'SUBSPACE', 'MIXED']),
			   "val_size": hp.uniform("val_size", 0.05, 0.25),
			   "epsilon": hp.uniform("epsilon", 0.05, 0.2),
			   "lr": hp.uniform("lr", 0.0001, 0.01),
			   "wd": hp.uniform("wd", 0.05, 0.2),
			   "n_epochs": hp.choice('n_epochs', np.arange(200, 500, dtype=int))}

# pelt_space = {"penalty": hp.quniform("penalty",5, 1000,1)}
pelt_space = {"penalty": hp.choice("penalty", np.arange(5, 500, dtype=int))}

bottomup_space = {"penalty": hp.choice("penalty", np.arange(5, 500, dtype=int))}


def get_layers(df):
	l1 = int(round(df.shape[1] / 4, 0))
	l2 = int(l1 * 2)
	l3 = int(l1 * 3)
	encoder = [l3, l2, l1]
	decoder = [l1, l2, l3]
	return encoder, decoder


# Validated Objectives
def knn_objective(space):
	'''
	KNN Hyperparameter Tuning Function
	:param space: hyperparameter space
	:return: accuracy
	'''
	# Instantiate the classifier
	clf = KNN(n_neighbors=space["n_neighbors"],
			  leaf_size=space["leaf_size"],
			  radius=space["radius"],
			  contamination=space["contamination"],
			  method=space["method"],
			  algorithm=space["algorithm"])
	clf.fit(X_train)
	pred = clf.predict(X_test)
	accuracy = f1_score(y_test, pred)
	return {'loss': -accuracy, 'status': STATUS_OK}


def vae_objective(space):
	'''
	VAE Hyperparameter Tuning Function
	:param space: hyperparameter space
	:return: accuracy
	'''
	encoder, decoder = get_layers(X_train)
	clf = VAE(encoder_neurons=encoder,
			  decoder_neurons=decoder,
			  # epochs = 1,
			  verbose=0,
			  gamma=space["gamma"],
			  validation_size=space["validation_size"],
			  capacity=space["capacity"],
			  batch_size=space["batch_size"],
			  # epochs=space["epochs"],
			  dropout_rate=space["dropout_rate"],
			  contamination=space["contamination"],
			  l2_regularizer=space["l2_regularizer"])
	clf.fit(X_train)
	pred = clf.predict(X_test)
	accuracy = f1_score(y_test, pred)
	return {'loss': -accuracy, 'status': STATUS_OK}


def deep_svdd_objective(space):
	'''
	Deep SVDD Hyperparameter Tuning Function
	:param space: hyperparameter space
	:return: accuracy
	'''
	encoder, decoder = get_layers(X_train)
	clf = DeepSVDD(hidden_neurons=encoder,
				   validation_size=space["validation_size"],
				   batch_size=space["batch_size"],
				   # epochs=space["epochs"],
				   dropout_rate=space["dropout_rate"],
				   contamination=space["contamination"],
				   l2_regularizer=space["l2_regularizer"])
	clf.fit(X_train)
	pred = clf.predict(X_test)
	accuracy = f1_score(y_test, pred)
	return {'loss': -accuracy, 'status': STATUS_OK}


def alad_objective(space):
	'''
	ALAD Hyperparameter Tuning Function
	:param space: hyperparameter space
	:return: accuracy
	'''
	
	encoder, decoder = get_layers(X_train)
	clf = ALAD(enc_layers=encoder,
			   dec_layers=decoder,
			   contamination=space["contamination"],
			   dropout_rate=space["dropout_rate"],
			   batch_size=space["batch_size"],
			   # epochs=space["epochs"],
			   learning_rate_gen=space["learning_rate_gen"],
			   learning_rate_disc=space["learning_rate_disc"])
	clf.fit(X_train)
	pred = clf.predict(X_test)
	accuracy = f1_score(y_test, pred)
	return {'loss': -accuracy, 'status': STATUS_OK}


def ecod_objective(space):
	'''
	ECOD Hyperparameter Tuning Function
	:param space: hyperparameter space
	:return: accuracy
	'''
	clf = ECOD(contamination=space["contamination"])
	clf.fit(X_train)
	pred = clf.predict(X_test)
	accuracy = f1_score(y_test, pred)
	return {'loss': -accuracy, 'status': STATUS_OK}


def abod_objective(space):
	'''
	ABOD Hyperparameter Tuning Function
	:param space: hyperparameter space
	:return: accuracy
	'''
	clf = ABOD(contamination=space["contamination"],
			   n_neighbors=space["n_neighbors"])
	clf.fit(X_train)
	pred = clf.predict(X_test)
	accuracy = f1_score(y_test, pred)
	return {'loss': -accuracy, 'status': STATUS_OK}


def lunar_objective(space):
	'''
	LUNAR Hyperparameter Tuning Function
	:param space: hyperparameter space
	:return: accuracy
	'''
	clf = LUNAR(model_type=space["model_type"],
				# n_neighbours=space["n_neighbours"],
				negative_sampling=space["negative_sampling"],
				val_size=space["val_size"],
				epsilon=space["epsilon"],
				lr=space["lr"],
				wd=space["wd"],
				# n_epochs = space["n_epochs"]
				)
	clf.fit(X_train)
	pred = clf.predict(X_test)
	accuracy = f1_score(y_test, pred)
	return {'loss': -accuracy, 'status': STATUS_OK}


def PELT_objective(space):
	'''
	PELT Hyperparameter Tuning Function
	:param space: hyperparameter space
	:return: accuracy
	'''
	clf = rpt.KernelCPD(kernel="rbf", min_size=2).fit(X_test.values)
	test_length = len(X_test)
	bkps = clf.predict(space["penalty"])
	# if the number of bkps is odd, remove the last one
	if len(bkps) % 2 != 0:
		bkps.pop()
	output = [0] * test_length
	# for each change point
	for i in range(0, len(bkps), 2):
		# mark the change point as anomaly
		for j in range(bkps[i], bkps[i + 1] + 1):
			if j < test_length:
				output[j] = 1
	accuracy = f1_score(y_test, output)
	return {'loss': -accuracy, 'status': STATUS_OK}


def bottomUp_objective(space):
	'''
	BottomUp Hyperparameter Tuning Function
	:param space: hyperparameter space
	:return: accuracy
	'''
	clf = rpt.detection.bottomup.BottomUp(model="rbf").fit(X_test.values)
	test_length = len(X_test)
	bkps = clf.predict(space["penalty"])
	if len(bkps) % 2 != 0:
		bkps.pop()
	output = [0] * test_length
	for i in range(0, len(bkps), 2):
		for j in range(bkps[i], bkps[i + 1] + 1):
			if j < test_length:
				output[j] = 1
	accuracy = f1_score(y_test, output)
	return {'loss': -accuracy, 'status': STATUS_OK}


def ChangePoint(X_test, penalty, model="PELT"):
    '''
    Change point detection using PELT or BottomUp
    :param X_test: time series
    :param penalty: penalty for change point
    :param model: PELT or BottomUp
    :return: change points

    Usage:
    cp = ChangePoint(X_test, penalty, model="PELT")
    '''
    if model == "PELT":
        algo = rpt.KernelCPD(kernel="rbf", min_size=2).fit(X_test.values)
    elif model == "BottomUp":
        algo = rpt.detection.bottomup.BottomUp(model="rbf").fit(X_test.values)
    else:
        return "Error: Please enter PELT or BottomUp"

    test_length = len(X_test)
    bkps = algo.predict(pen=penalty)

    if len(bkps) % 2 != 0:
        bkps.pop()
    output = [0] * test_length
    for i in range(0, len(bkps), 2):
        for j in range(bkps[i], bkps[i + 1] + 1):
            if j < test_length:
                output[j] = 1
            else:
                return output
    return output




def save_hyper_params(path, p, model_name):
	with open(path, 'a', newline='') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(["Model", model_name])
		for key, value in p.items():
			writer.writerow([key, value])

# save_hyper_params("/content/hyper_params", {'penalty': 3171.0}, 'PELT')

evals = 25
path = ""


trials = Trials()
param_lunar = fmin(fn=lunar_objective,
            space=lunar_space,
            algo=tpe.suggest,
            max_evals=evals,
            trials=trials)

save_hyper_params(path, param_lunar, 'LUNAR')

trials = Trials()
param_abod = fmin(fn=abod_objective,
            space=abod_space,
            algo=tpe.suggest,
            max_evals=evals,
            trials=trials)
save_hyper_params(path, param_abod, 'ABOD')


trials = Trials()
param_ecod = fmin(fn=ecod_objective,
            space=ecod_space,
            algo=tpe.suggest,
            max_evals=evals,
            trials=trials)

save_hyper_params(path, param_ecod, 'ECOD')

trials = Trials()
param_alad = fmin(fn=alad_objective,
            space=alad_space,
            algo=tpe.suggest,
            max_evals=evals,
            trials=trials)

save_hyper_params(path, param_alad, 'ALAD')

trials = Trials()
param_deep_svdd = fmin(fn=deep_svdd_objective,
            space=deep_svdd_space,
            algo=tpe.suggest,
            max_evals=evals,
            trials=trials)
save_hyper_params(path, param_deep_svdd, 'DeepSVDD')

trials = Trials()
param_vae = fmin(fn=vae_objective,
            space=vae_space,
            algo=tpe.suggest,
            max_evals=evals,
            trials=trials)
save_hyper_params(path, param_vae, 'VAE')

trials = Trials()
param_knn = fmin(fn=knn_objective,
            space=knn_space,
            algo=tpe.suggest,
            max_evals=evals,
            trials=trials)
save_hyper_params(path, param_knn, 'KNN')

trials = Trials()

param_pelt = fmin(fn=PELT_objective,
            space=pelt_space,
            algo=tpe.suggest,
            max_evals=25,
            trials=trials)
save_hyper_params(path, param_pelt, 'PELT')


trials = Trials()
param_bottomup = fmin(fn=bottomUp_objective,
            space=bottomup_space,
            algo=tpe.suggest,
            max_evals=evals,
            trials=trials)
save_hyper_params(path, param_bottomup, 'Bottom Up')
