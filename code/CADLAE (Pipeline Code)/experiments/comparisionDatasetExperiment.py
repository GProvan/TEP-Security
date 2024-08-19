from pyod.models.ecod import ECOD
from pyod.models.abod import ABOD
from pyod.models.lunar import LUNAR
from pyod.models.vae import VAE
from pyod.models.alad import ALAD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.knn import KNN
import ruptures as rpt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from joblib import dump, load
penalty = 50

import torch
from tqdm import trange
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

sys.path.append('../')
from cadlae.preprocessComparison import *
from cadlae.detector import *


def get_layers(df):
	l1 = int(round(df.shape[1] / 4, 0))
	l2 = int(l1 * 2)
	l3 = int(l1 * 3)
	encoder = [l3, l2, l1]
	decoder = [l1, l2, l3]
	return encoder, decoder


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


def _results(accuracy, pre, rec, f1, roc):
	'''
	Function to calculate the results
	:param accuracy: accuracy score
	:param pre: precision score
	:param rec: recall score
	:param f1: f1 score
	:param roc: roc_auc score
	:return: accuracy, precision, recall, f1, roc_auc

	Usage:
	>> acc,precision, recall, f1_score, roc_auc = _results(accuracy, precision, recall , f1, roc)
	'''
	acc = round(accuracy * 100, 2)
	precision = round(pre * 100, 2)
	recall = round(rec * 100, 2)
	f1_score = round(f1 * 100, 2)
	roc_auc = round(roc * 100, 2)
	return [acc, precision, recall, f1_score, roc_auc]


def calculate_results_return(y_test, y_pred):
	'''
	Function to calculate the results
	:param y_test: test data
	:param y_pred: predicted data
	:return: accuracy, precision, recall, f1, roc_auc

	Usage:
	>> acc,precision, recall, f1_score, roc_auc = calculate_results(y_test, y_pred)
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
	acc, precision, recall, f1_score, roc_auc = _results(accuracy, precision, recall, f1, roc)
	return [acc, precision, recall, f1_score, roc_auc]


def result_dataframe(result_dict, y_test, fault):
	'''
	Function to create a dataframe of the results
	:param result_dict: dictionary of the results
	:param y_test: test data
	:param fault: fault type
	:return: dataframe of the results

	Usage:
	>> df = result_dataframe(result_dict, y_test, fault)
	'''
	df = pd.DataFrame()
	for model in result_dict:
		res = calculate_results_return(y_test, result_dict[model])
		df = df.append({"Fault": fault, 'Model': model, 'Accuracy': res[0], 'Precision': res[1], 'Recall': res[2],
						'F1 Score': res[3], 'ROCAUC': res[4]}, ignore_index=True)
	return df


def join_df(df_list):
	'''
	Function to join the dataframes
	:param df_list: list of dataframes
	:return: joined dataframe

	Usage:
	>> df = join_df(df_list)
	'''
	df = pd.concat(df_list, ignore_index=True)
	return df


def fault_name(link):
	'''
	Function to get the fault name
	:param link: link to the fault
	:return: fault name

	Usage:
	>> fault = fault_name(link)
	'''
	parts = link.split('/')
	filename = parts[-1]  # get the last part of the split string, which is the file name
	idv = filename.split('_')[-1].split('.')[0]  # get the last part before the '.'
	return idv


def convert_dict_to_df(dictionary, y_true, fault):
	'''
	Function to convert the dictionary to a dataframe
	:param dictionary: dictionary of the results
	:param y_true: true labels
	:param fault: fault name
	:return: dataframe of the results

	Usage:
	>> df = convert_dict_to_df(dictionary,y_true, fault)
	'''
	dictionary["y_true"] = y_true
	df = pd.DataFrame.from_dict(dictionary, orient='index').transpose()
	df["Fault"] = fault
	return df


from joblib import dump, load


def dump_model(path, model, name):
	dump(model, '/{}/{}.joblib'.format(path, name))


def load_model(path, name):
	return load('/{}/{}.joblib'.format(path, name))


def find_optimal_threshold(y_test, y_pred):
	'''
	:param y_test: test target variable
	:param y_pred: predicted target variable
	:return: optimal threshold
	'''
	# get the false positive rate, true positive rate and thresholds
	fpr, tpr, thresholds = roc_curve(y_test, y_pred)
	# get the optimal threshold based on the maximum tpr - fpr
	optimal_idx = np.argmax(tpr - fpr)
	# get the optimal threshold
	optimal_threshold = thresholds[optimal_idx]
	return optimal_threshold


def convert_scores_to_label(array, threshold):
	'''
	The function converts the scores to labels based on the threshold, if the score is greater than the threshold,
	the label is 1, else 0
	:param array: array of scores
	:param threshold: threshold to use for converting the scores to labels
	:return: array of labels
	'''
	binary = []
	for i in array:
		if i < threshold:
			binary.append(0)
		else:
			binary.append(1)
	return binary

hyper_params = {'shuttle': {"ECOD": {'contamination': 0.059028668544143376},
							"ABOD": {'contamination': 0.08836912569400507,
									 "n_neighbors": 14},
							"LUNAR": {'epsilon': 0.18850494601981743,
									  'lr': 0.007207762368358249,
									  'model_type': "WEIGHT",
									  'n_neighbors': 3,
									  'negative_sampling': 'UNIFORM',
									  'val_size': 0.13210554520236506,
									  'wd': 0.1800767411452292},
							"ALAD": {'batch_size': 32,
									 'contamination': 0.11805157742432482,
									 'dropout_rate': 0.3038442725712914,
									 'learning_rate_disc': 0.0021112800464264813,
									 'learning_rate_gen': 0.009268324104567759},
							"DeepSVDD": {'batch_size': 128,
										 'contamination': 0.18744213343043975,
										 'dropout_rate': 0.130364309365826,
										 'l2_regularizer': 0.2503415680477573,
										 'validation_size': 0.15147304547858176},
							'KNN': {'contamination': 0.08836912569400507,
									'algorithm': "auto",
									'leaf_size': 30,
									'method': "median",
									'n_neighbors': 16,
									'radius': 0.9078634037268895},

							'VAE': {'batch_size': 64,
									'contamination': 0.08836912569400507,
									'capacity': 0.0920496315189437,
									'dropout_rate': 0.382919881732925,
									'gamma': 1.5030396851188994,
									'l2_regularizer': 0.4228984319897774,
									'validation_size': 0.14117091863231462},
							"PROPOSED": {'num_epochs': 10,
										 'batch_size': 64,
										 'lr': 1e-3,
										 'hidden_size': 3,
										 'sequence_length': 30,
										 'train_gaussian_percentage': 0.25,
										 'n_layers': 1,
										 'use_bias': True,
										 'dropout': 0.2}},
				'mammography': {"ECOD": {'contamination': 0.05012044410097263},
								"ABOD": {'contamination': 0.13323622976536637,
										 "n_neighbors": 23},
								"LUNAR": {'epsilon': 0.08904160938494599,
										  'lr': 0.009084520338946661,
										  'model_type': "SCORE",
										  'n_neighbors': 11,
										  'negative_sampling': 'UNIFORM',
										  'val_size': 0.1836202950387711,
										  'wd': 0.14609755824820392},
								"ALAD": {'batch_size': 256,
										 'contamination': 0.07767407191669636,
										 'dropout_rate': 0.6259529109411545,
										 'learning_rate_disc': 0.005612384916525416,
										 'learning_rate_gen': 0.0025121608162788247},
								"DeepSVDD": {'batch_size': 256,
											 'contamination': 0.08148905179311877,
											 'dropout_rate': 0.39299060534500296,
											 'l2_regularizer': 0.1878562323179467,
											 'validation_size': 0.07833378225189457},
								'KNN': {'contamination': 0.0504061,
										'algorithm': "auto",
										'leaf_size': 30,
										'method': "median",
										'n_neighbors': 16,
										'radius': 0.9078634037268895},
								'VAE': {'batch_size': 217,
										'contamination': 0.05038545710467858,
										'capacity': 0.10820708594120484,
										'dropout_rate': 0.19989364408707838,
										'gamma': 2.417045890440947,
										'l2_regularizer': 0.12746699380674167,
										'validation_size': 0.1442991903650632},
								"PROPOSED": {'num_epochs': 10,
											 'batch_size': 32,
											 'lr': 1e-3,
											 'hidden_size': 2,
											 'sequence_length': 30,
											 'train_gaussian_percentage': 0.25,
											 'n_layers': 1,
											 'use_bias': True,
											 'dropout': 0.2}},
				'cover': {"ECOD": {'contamination': 0.08843824496605635},
						  "ABOD": {'contamination': 0.06473607368780376,
								   "n_neighbors": 16},
						  "LUNAR": {'epsilon': 0.07232738903321538,
									'lr': 0.0032501216292100893,
									'model_type': "WEIGHT",
									'n_neighbors': 16,
									'negative_sampling': "SUBSPACE",
									'val_size': 0.11806319758272506,
									'wd': 0.15629265858646108},
						  "ALAD": {'batch_size': 96,
								   'contamination': 0.05267821556138073,
								   'dropout_rate': 0.16087943190711362,
								   'learning_rate_disc': 0.008701465090680188,
								   'learning_rate_gen': 0.0039202509441753795},
						  "DeepSVDD": {'batch_size': 164,
									   'contamination': 0.07078234132189587,
									   'dropout_rate': 0.15981520354721937,
									   'l2_regularizer': 0.4344044226783663,
									   'validation_size': 0.08187612618569545},
						  'KNN': {'contamination': 0.052685810203913164,
								  'algorithm': "auto",
								  'leaf_size': 30,
								  'method': "median",
								  'n_neighbors': 19,
								  'radius': 0.9078634037268895},
						  'VAE': {'batch_size': 32,
								  'contamination': 0.050822836505112545,
								  'capacity': 0.06328952478924739,
								  'dropout_rate': 0.3453389796735339,
								  'gamma': 2.1408179340734383,
								  'l2_regularizer': 0.6301009158751907,
								  'validation_size': 0.10117091863231462},
						  "PROPOSED": {'num_epochs': 10,
									   'batch_size': 256,
									   'lr': 1e-3,
									   'hidden_size': 5,
									   'sequence_length': 30,
									   'train_gaussian_percentage': 0.25,
									   'n_layers': 1,
									   'use_bias': True,
									   'dropout': 0.2}},
				'campaign': {"ECOD": {'contamination': 0.11627399513020334},
							 "ABOD": {'contamination': 0.10644666880999723,
									  "n_neighbors": 27},
							 "LUNAR": {'epsilon': 0.1067517224838724,
									   'lr': 0.0036694719776562603,
									   'model_type': "SCORE",
									   'n_neighbors': 5,
									   'negative_sampling': "SUBSPACE",
									   'val_size': 0.12326412859290242,
									   'wd': 0.14630222946562016},
							 "ALAD": {'batch_size': 96,
									  'contamination': 0.11643548543716847,
									  'dropout_rate': 0.5499992156952807,
									  'learning_rate_disc': 0.008719813322307493,
									  'learning_rate_gen': 0.008790477929875896},
							 "DeepSVDD": {'batch_size': 164,
										  'contamination': 0.15554828353952876,
										  'dropout_rate': 0.14038475702074868,
										  'l2_regularizer': 0.3185554032105434,
		
										  'validation_size': 0.14661082745000747},
							 'KNN': {'contamination': 0.12140189229827425,
									 'algorithm': "auto",
									 'leaf_size': 40,
									 'method': "median",
									 'n_neighbors': 19,
									 'radius': 1.1463166262536306},
							 'VAE': {'batch_size': 164,
									 'contamination': 0.11764945287767577,
									 'capacity': 0.0920496315189437,
									 'dropout_rate': 0.382919881732925,
									 'gamma': 1.5030396851188994,
									 'l2_regularizer': 0.4228984319897774,
									 'validation_size': 0.1423806356336121},
							 "PROPOSED": {'num_epochs': 10,
										  'batch_size': 64,
										  'lr': 1e-3,
										  'hidden_size': 30,
										  'sequence_length': 30,
										  'train_gaussian_percentage': 0.25,
										  'n_layers': 1,
										  'use_bias': True,
										  'dropout': 0.2}},
				'thyroid': {"ECOD": {'contamination': 0.05090912869246383},
							"ABOD": {'contamination': 0.053790177649502224,
									 "n_neighbors": 17},
							"LUNAR": {'epsilon': 0.06349698135890142,
									  'lr': 0.0057834517932312856,
									  'model_type': "SCORE",
									  'n_neighbors': 6,
									  'negative_sampling': "UNIFORM",
									  'val_size': 0.1310993763328278,
									  'wd': 0.09809801857790652},
							"ALAD": {'batch_size': 132,
									 'contamination': 0.16308764374643292,
									 'dropout_rate': 0.5457534182482985,
									 'learning_rate_disc': 0.009566845995068319,
									 'learning_rate_gen': 0.008098944395007992},
							"DeepSVDD": {'batch_size': 96,
										 'contamination': 0.1473261557656635,
										 'dropout_rate': 0.32028838209701327,
										 'l2_regularizer': 0.18157298005590206,
										 'validation_size': 0.09424926260813332},
							'KNN': {'contamination': 0.076821576925705,
									'algorithm': "auto",
									'leaf_size': 25,
									'method': "median",
									'n_neighbors': 14,
									'radius': 1.1227189321502937},
							'VAE': {'batch_size': 128,
									'contamination': 0.05799219281014927,
									'capacity': 0.12315663074925337,
									'dropout_rate': 0.3486584887949361,
									'gamma': 1.259613792616077,
									'l2_regularizer': 0.4205177215764463,
									'validation_size': 0.14256793208658947},
							"PROPOSED": {'num_epochs': 10,
										 'batch_size': 64,
										 'lr': 0.001,
										 'hidden_size': 3,
										 'sequence_length': 30,
										 'train_gaussian_percentage': 0.25,
										 'n_layers': 1,
										 'use_bias': True,
										 'dropout': 0.2}},
				'satimage-2': {"ECOD": {'contamination': 0.07480003963208715},
							   "ABOD": {'contamination': 0.09300471879997917,
										"n_neighbors": 18},
							   "LUNAR": {'epsilon': 0.07828474121347008,
										 'lr': 0.0016241032137395894,
										 'model_type': "SCORE",
										 'n_neighbors': 8,
										 'negative_sampling': "UNIFORM",
										 'val_size': 0.10088318081649571,
										 'wd': 0.10229936875578807},
							   "ALAD": {'batch_size': 164,
										'contamination': 0.05174899529157598,
										'dropout_rate': 0.5130409813060697,
										'learning_rate_disc': 0.00852030131349943,
										'learning_rate_gen': 0.007033115983782491},
							   "DeepSVDD": {'batch_size': 96,
											'contamination': 0.08119675381363552,
											'dropout_rate': 0.3052615383207474,
											'l2_regularizer': 0.19057723986208974,
											'validation_size': 0.1620674689669866},
							   'KNN': {'contamination': 0.060565753139506406,
									   'algorithm': "auto",
									   'leaf_size': 22,
									   'method': "median",
									   'n_neighbors': 17,
									   'radius': 1.1998897220962816},
							   'VAE': {'batch_size': 96,
									   'contamination': 0.05329288919661715,
									   'capacity': 0.023794163965108518,
									   'dropout_rate': 0.3664012982086703,
									   'gamma': 1.1673743249862047,
									   'l2_regularizer': 0.4047699568444111,
									   'validation_size': 0.092211656751815},
							   "PROPOSED": {'num_epochs': 10,
											'batch_size': 32,
											'lr': 1e-3,
											'hidden_size': 18,
											'sequence_length': 30,
											'train_gaussian_percentage': 0.25,
											'n_layers': 1,
											'use_bias': True,
											'dropout': 0.2}},
				'annthyroid': {"ECOD": {'contamination': 0.06624658763051512},
							   "ABOD": {'contamination': 0.09154603066936562,
										"n_neighbors": 18},
							   "LUNAR": {'epsilon': 0.09744093026731726,
										 'lr': 0.002651046297902741,
										 'model_type': "SCORE",
										 'n_neighbors': 10,
										 'negative_sampling': "UNIFORM",
										 'val_size': 0.057313915502969916,
										 'wd': 0.16429674317521545},
							   "ALAD": {'batch_size': 196,
										'contamination': 0.1608140201446709,
										'dropout_rate': 0.2522754777744472,
										'learning_rate_disc': 0.002284228857542532,
										'learning_rate_gen': 0.0005691803722820053},
							   "DeepSVDD": {'batch_size': 96,
											'contamination': 0.06678754376758347,
											'dropout_rate': 0.29003786943788146,
											'l2_regularizer': 0.6774451981105774,
											'validation_size': 0.05105415898462459},
							   'KNN': {'contamination': 0.10516974141687994,
									   'algorithm': 'ball_tree',
									   'leaf_size': 22,
									   'method': "mean",
									   'n_neighbors': 9,
									   'radius': 0.6251971811569987},
							   'VAE': {'batch_size': 132,
									   'contamination': 0.06328618952612175,
									   'capacity': 0.08148198527590088,
									   'dropout_rate': 0.12508149190878928,
									   'gamma': 2.17838878145976,
									   'l2_regularizer': 0.3234522989105527,
									   'validation_size': 0.10010713036804711},
							   "PROPOSED": {'num_epochs': 10,
											'batch_size': 32,
											'lr': 1e-3,
											'hidden_size': 3,
											'sequence_length': 30,
											'train_gaussian_percentage': 0.25,
											'n_layers': 1,
											'use_bias': True,
											'dropout': 0.2}}
				}


def run(dataset_name, X_train, X_test, y_test, path_out, train_models=True, save_models=False, path_models=None):
	def model_init(train_models):
		encoder, decoder = get_layers(X_train)
		models = {}
		if train_models == False and path_models == None:
			print("Training is False, however no path to Models is provided")
		
		
		elif train_models == True:
			ecod = ECOD(contamination=hyper_params[dataset_name]["ECOD"]["contamination"])
			abod = ABOD(contamination=hyper_params[dataset_name]["ABOD"]["contamination"],
						n_neighbors=hyper_params[dataset_name]["ABOD"]["n_neighbors"])
			lunar = LUNAR(epsilon=hyper_params[dataset_name]["LUNAR"]["epsilon"],
						  lr=hyper_params[dataset_name]["LUNAR"]["lr"],
						  model_type=hyper_params[dataset_name]["LUNAR"]["model_type"],
						  n_neighbours=hyper_params[dataset_name]["LUNAR"]["n_neighbors"],
						  negative_sampling=hyper_params[dataset_name]["LUNAR"]["negative_sampling"],
						  val_size=hyper_params[dataset_name]["LUNAR"]["val_size"],
						  wd=hyper_params[dataset_name]["LUNAR"]["wd"])
			alad = ALAD(enc_layers=encoder,
						dec_layers=decoder,
						batch_size=hyper_params[dataset_name]["ALAD"]["batch_size"],
						contamination=hyper_params[dataset_name]["ALAD"]["contamination"],
						dropout_rate=hyper_params[dataset_name]["ALAD"]["dropout_rate"],
						learning_rate_disc=hyper_params[dataset_name]["ALAD"]["learning_rate_disc"],
						learning_rate_gen=hyper_params[dataset_name]["ALAD"]["learning_rate_gen"])
			
			deepsvdd = DeepSVDD(hidden_neurons=encoder,
								batch_size=hyper_params[dataset_name]["DeepSVDD"]["batch_size"],
								contamination=hyper_params[dataset_name]["DeepSVDD"]["contamination"],
								dropout_rate=hyper_params[dataset_name]["DeepSVDD"]["dropout_rate"],
								l2_regularizer=hyper_params[dataset_name]["DeepSVDD"]["l2_regularizer"],
								validation_size=hyper_params[dataset_name]["DeepSVDD"]["validation_size"])
			
			knn = KNN(contamination=hyper_params[dataset_name]["KNN"]["contamination"],
					  algorithm=hyper_params[dataset_name]["KNN"]["algorithm"],
					  leaf_size=hyper_params[dataset_name]["KNN"]["leaf_size"],
					  radius=hyper_params[dataset_name]["KNN"]["radius"],
					  method=hyper_params[dataset_name]["KNN"]["method"],
					  n_neighbors=hyper_params[dataset_name]["KNN"]["n_neighbors"])
			
			vae = VAE(encoder_neurons=encoder,
					  decoder_neurons=decoder,
					  batch_size=hyper_params[dataset_name]["VAE"]["batch_size"],
					  contamination=hyper_params[dataset_name]["VAE"]["contamination"],
					  capacity=hyper_params[dataset_name]["VAE"]["capacity"],
					  dropout_rate=hyper_params[dataset_name]["VAE"]["dropout_rate"],
					  gamma=hyper_params[dataset_name]["VAE"]["gamma"],
					  l2_regularizer=hyper_params[dataset_name]["VAE"]["l2_regularizer"],
					  validation_size=hyper_params[dataset_name]["VAE"]["validation_size"])
			
			proposed = AnomalyDetector(num_epochs=hyper_params[dataset_name]["PROPOSED"]["num_epochs"],
									   batch_size=hyper_params[dataset_name]["PROPOSED"]["batch_size"],
									   lr=hyper_params[dataset_name]["PROPOSED"]["lr"],
									   hidden_size=hyper_params[dataset_name]["PROPOSED"]["hidden_size"],
									   sequence_length=hyper_params[dataset_name]["PROPOSED"]["sequence_length"],
									   train_gaussian_percentage=hyper_params[dataset_name]["PROPOSED"][
										   "train_gaussian_percentage"],
									   n_layers=hyper_params[dataset_name]["PROPOSED"]["n_layers"],
									   use_bias=hyper_params[dataset_name]["PROPOSED"]["use_bias"],
									   dropout=hyper_params[dataset_name]["PROPOSED"]["dropout"])
			
			pelt = rpt.KernelCPD(kernel="rbf", min_size=2)
			bottomup = rpt.detection.bottomup.BottomUp(model="rbf")
			
			models["ECOD"] = ecod
			models["ABOD"] = abod
			models['LUNAR'] = lunar
			models['VAE'] = vae
			models['ALAD'] = alad
			models['DeepSVDD'] = deepsvdd
			models['KNN'] = knn
			models['PROPOSED'] = proposed
			models['PELT'] = pelt
			models['BottomUp'] = bottomup
		
		
		
		elif train_models == False:
			ecod = load(path_models + 'ECOD.joblib')
			abod = load(path_models + 'ABOD.joblib')
			lunar = load(path_models + 'LUNAR.joblib')
			alad = load(path_models + 'ALAD.joblib')
			deepsvdd = load(path_models + 'DeepSVDD.joblib')
			knn = load(path_models + 'KNN.joblib')
			vae = load(path_models + 'VAE.joblib')
			proposed = AnomalyDetector()
			proposed = torch.load(path_models + 'PROPOSED.pth')
			# Change Point dont do training
			pelt = rpt.KernelCPD(kernel="rbf", min_size=2)
			bottomup = rpt.detection.bottomup.BottomUp(model="rbf")
			models["ECOD"] = ecod
			models["ABOD"] = abod
			models['LUNAR'] = lunar
			models['VAE'] = vae
			models['ALAD'] = alad
			models['DeepSVDD'] = deepsvdd
			models['KNN'] = knn
			models['PROPOSED'] = proposed
			models['PELT'] = pelt
			models['BottomUp'] = bottomup
		
		return models
	
	def train(X_train, model_dict, save_models=True, directory=None, train_list=None, skip_list=None):
		trained = {}
		if train_list != None:
			models_to_train = train_list
		else:
			models_to_train = ["ECOD", "ABOD", "LUNAR", "VAE", "ALAD", "DeepSVDD", "KNN", "PROPOSED"]
		
		if skip_list != None:
			models_to_skip = skip_list
		else:
			models_to_skip = ['PELT', 'BottomUp']
		
		for name, model in model_dict.items():
			if name in models_to_train:
				print("TRAINING MODEL: {}".format(name))
				model_dict[name].fit(X_train)
				if save_models == True:
					if name == "PROPOSED":
						torch.save(model_dict[name], str(directory + "proposed.pth"))
					else:
						dump(model_dict[name], str(directory + '{}.joblib'.format(name)))
				
				trained[name] = model_dict[name]
			# just reasign the CP model
			elif name in models_to_skip:
				trained[name] = model_dict[name]
		
		return trained
	
	def test(trained_models, X_test, y_test):
		pyod = ["ECOD", "ABOD", "LUNAR", "VAE", "ALAD", "DeepSVDD", "KNN"]
		prop = ["PROPOSED"]
		changepoint = ['PELT', 'BottomUp']
		results = {}
		for model in trained_models:
			print("TESTING MODEL: {}".format(model))
			if model in pyod:
				y_pred = trained_models[model].predict(X_test)
				results[model] = list(y_pred)
			elif model in prop:
				yp, details = trained_models[model].predict(X_test)
				optimal = find_optimal_threshold(y_test, yp)
				# convert the scores to labels
				y_pred = convert_scores_to_label(yp, optimal)
				results[model] = y_pred
			'''
			else:
				test_length = len(X_test)
				trained_models[model].fit(X_test.values)
				bkps = models[model].predict(pen=penalty)
				if len(bkps) % 2 != 0:
					bkps.pop()
				y_pred = [0] * test_length
				for i in range(0, len(bkps), 2):
					for j in range(bkps[i], bkps[i + 1] + 1):
						if j < test_length:
							y_pred[j] = 1

				results[model] = y_pred
			'''
		
		return results
	
	models = model_init(train_models)
	trained = train(X_train, models, True, "/content/")
	result_dict = test(trained, X_test, y_test)
	predictions = result_dict
	predictions["y_true"] = list(y_test)
	result = result_dataframe(result_dict, y_test, "Fault")
	result.to_csv(path_out + "result_table_{}.csv".format(dataset_name))
	return result, predictions

path = ""
X_train, y_train, X_test, y_test, col_names = preprocess_datasets("adbench","/content/2_annthyroid.npz",0.1 )
result,prediction = run("annthyroid",X_train,X_test,y_test,"/content/", True,True)

result.to_csv(path + "results.csv")
