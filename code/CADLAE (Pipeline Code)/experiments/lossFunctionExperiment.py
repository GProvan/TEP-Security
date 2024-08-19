import sys
import torch
import pandas as pd
import numpy as np
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
from pyod.models.auto_encoder_torch import AutoEncoder
from sklearn.metrics import roc_auc_score
# add the parent directory to the path
sys.path.append('../')
from cadlae.customLoss import CosineLoss, PrincipalAngleLoss, OneClassSVMLoss
from cadlae.preprocessTEP import *


# load the data, change the path to the data file
train_link = "/content/drive/My Drive/TEPdata/normal_10000.csv"
test_link = "/content/drive/My Drive/TEPdata/df_test_idv4_10000.csv"
processor = DataProcessor(train_link, test_link, "Fault", "Unnamed: 0")
X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test
scaler = processor.scaler_function
col_names = processor.col_names

# Create the model
cosine_ae = AutoEncoder(loss_fn = CosineLoss(),epochs = 100,hidden_neurons = [52,26,13],batch_size = 64,preprocessing  = False)
mse_ae = AutoEncoder(epochs = 100,hidden_neurons = [52,26,13],batch_size = 64,preprocessing  = False)
pa_ae = AutoEncoder(loss_fn = PrincipalAngleLoss(),epochs = 100,hidden_neurons = [52,26,13],batch_size = 64,preprocessing  = False)
ocsvm_ae = AutoEncoder(loss_fn = OneClassSVMLoss(),epochs = 100,hidden_neurons = [52,26,13],batch_size = 64,preprocessing  = False)


# Fit the models
cosine_ae.fit(X_train)
mse_ae.fit(X_train)
pa_ae.fit(X_train)
ocsvm_ae.fit(X_train)

# Get the Predictions
pred_cosine = cosine_ae.predict(X_test)
pred_mse = mse_ae.predict(X_test)
pred_pa = pa_ae.predict(X_test)
pred_ocsvm = ocsvm_ae.predict(X_test)


# Get the AUC
auc_cosine = roc_auc_score(y_test,pred_cosine)
auc_mse = roc_auc_score(y_test,pred_mse)
auc_pa = roc_auc_score(y_test,pred_pa)
auc_ocsvm = roc_auc_score(y_test,pred_ocsvm)
print('ROC-AUC Cosine: ' + str(auc_cosine))
print('ROC-AUC MSE: ' + str(auc_mse))
print('ROC-AUC PA: ' + str(auc_pa))
print('ROC-AUC OCSVM: ' + str(auc_ocsvm ))