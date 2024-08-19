
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
import torch
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import logging
import abc
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
from lightgbm import LGBMClassifier
import warnings
from collections import Counter
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


sys.path.append('../')
from cadlae.detector import *
from cadlae.preprocessTEP import *
from CorrelationSubgraph import *
from localisationFeatureWise import *
from localisationSubgraph import *


train_link = "/content/drive/My Drive/TEPdata/experiment_1/normal_10000.csv"
test_link = "/content/drive/My Drive/TEPdata/experiment_1/df_IDV(5).csv"
model_link = "/content/b256_e10000_h25.pth"
processor = DataProcessor(train_link, test_link, "Fault", "Unnamed: 0")
X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test
scaler = processor.scaler_function

model = AnomalyDetector()
model= torch.load('/Users/cameronlooney/PycharmProjects/CADLAE/venv/cadlae/CADLAE_model.pth')

model_train = AnomalyDetector()
model_train.fit(X_train)
t_scores,d_train = model_train.predict(X_train)
train_scores,details_train= t_scores.copy(),d_train.copy()


def localisation_pipeline(test_data, start_index,end_index,training_error = None):
    processor = DataProcessor(train_link, test_data, "Fault", "Unnamed: 0")
    X_train = processor.X_train
    y_train = processor.y_train
    X_test = processor.X_test[start_index:end_index]
    y_test = processor.y_test[start_index:end_index]
    scaler = processor.scaler_function

    # new model to get training errors
    if training_error ==None:
        model_train = AnomalyDetector()
        model_train.fit(X_train)
        t_scores,d_train = model_train.predict(X_train)
        train_scores,details_train= t_scores.copy(),d_train.copy()
    else:
        details_train = training_error


    # predict with pre-trained model
    test_preds, details_test = model.predict(X_test)

    # Feature wise localisation based on training reconstruction error
    ftwise= FeatureWiseLocalisation(y_test, test_preds, processor.col_names, details_train, details_test)
    rank,y_predictions = ftwise.run()

    # Build Correlation Graph
    subgrph = CorrelationSubgraph(X_train,0.6)
    subgraph_dict = subgrph.generate_subgraph_dict()

    # Localise Subgraph
    local = LocaliseSubgraph(rank, subgraph_dict)
    likely_subgraph = local.find_max_subgraph()

    # Rank features in subgraph
    try:
        localisation_report = local.rank_subgraph(likely_subgraph)
    except:
        localisation_report = None
    return localisation_report

def generate_interval_lists(n):
    interval_lists = []
    start = 675
    end = start + 10
    i = 1
    while i <= n:
        interval_lists.append([start, end])
        start = end + 675
        end = start + (10+ 10*i)
        i+=1
    return interval_lists


test_link = "/content/drive/My Drive/TEPdata/experiment_3/local_IDV(4).csv"
#test_link = "/content/drive/My Drive/TEPdata/experiment_1/df_IDV(17).csv"
#report = localisation_pipeline(test_link,0,10000,details_train)
def check(anom,test_link,details_train):
    counter = 1
    for i in anom:
        start = i[0] - 300
        end = i[1] + 300
        print("Anomaly {}/15".format(counter))
        localisation_pipeline(test_link,start,end,details_train)
        counter+=1

check(anom, test_link, details_train)
