import torch
from tqdm import trange
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import pandas as pd
import torch.nn as nn
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve,roc_auc_score


def set_seed(seed=None):
	'''
	Set seed for all libraries
	:param seed: seed
	'''
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
	else:
		pass


class AnomalyDetector():
	def __init__(self, num_epochs: int = 5, batch_size: int = 20, lr: float = 1e-3,
				 hidden_size: int = 25, sequence_length: int = 30, train_gaussian_percentage: float = 0.25,
				 n_layers: int = 1, use_bias: bool = True, dropout: float = 0.2):
		
		'''
		The constructor takes in a number of arguments and sets them as instance variables.
		The instance variables include hyperparameters for training an LSTM model such as the number of epochs,
		batch size, and learning rate, as well as model architecture parameters such as the hidden size,
		number of layers, and dropout rate. The class also has several instance variables for storing the
		trained model, mean, and covariance of a Gaussian distribution. The LSTMED class also has a device
		instance variable which is set to "cuda:0" if a GPU is available and "cpu" otherwise. The LSTMED
		class also has a seed instance variable which is used to set the random seeds for Python's built-in
		random module, NumPy's random module, and PyTorch's random module. This can be useful for reproducing
		results. The LSTMED class also has a details instance variable which is a boolean indicating whether
		or not to store prediction details. Finally, the LSTMED class has an instance variable called lstmed
		which will be used to store the trained LSTM model.

		:param name: name of the model
		:param num_epochs: number of epochs to train the model
		:param batch_size: batch size to use for training the model
		:param lr: learning rate to use for training the model
		:param hidden_size: number of hidden units in the LSTM model
		:param sequence_length: number of time steps in the LSTM model
		:param train_gaussian_percentage: percentage of training data to use for training the Gaussian distribution
		:param n_layers: number of layers in the LSTM model
		:param use_bias: whether or not to use bias in the LSTM model
		:param dropout: dropout rate in the LSTM model
		:param seed: random seed to use for reproducibility
		:param gpu: GPU to use for training the model
		:param details: whether or not to store prediction details
		'''
		
		# set the random seed
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.prediction_details = {}
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.lr = lr
		self.hidden_size = hidden_size
		self.sequence_length = sequence_length
		self.train_gaussian_percentage = train_gaussian_percentage
		self.n_layers = n_layers
		self.use_bias = use_bias
		self.dropout = dropout
		
		# will be used to store the trained LSTM model
		self.lstmed = None
		# will be used to store the mean and covariance of the Gaussian distribution
		self.mean, self.cov = None, None
	
	def to_device(self, model):
		'''
		Move the model to the device specified by the user
		:param model: model to move to the device
		'''
		model.to(self.device)
	
	def fit(self, X: pd.DataFrame):
		'''
		Define's a method fit the model, takes as input a pandas DataFrame X.
		The method begins by converting X to a NumPy array and then creating a list of sub-arrays,
		each of which is a sequence of self.sequence_length consecutive rows of data.
		These sequences are then randomly shuffled and split into two subsets, a training set and a
		"train Gaussian" set. The training set is used to train an LSTM-based module called self.lstmed
		using an Adam optimizer, while the "train Gaussian" set is used to evaluate the model.
		Finally, the mean and covariance of the error between the model's predictions and the actual
		values are computed for the "train Gaussian" set and stored as attributes self.mean and self.cov,
		respectively.

		:param X: pandas DataFrame to fit the model
		:return: None
		'''
		data = X.values
		# create sequences of data by taking self.sequence_length consecutive rows
		sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
		# shuffle the sequences
		indices = np.random.permutation(len(sequences))
		# split the sequences into a training set and a "train Gaussian" set
		split_point = int(self.train_gaussian_percentage * len(sequences))
		# training set
		train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
								  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
		train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
										   sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)
		
		# create the LSTM model using the DetectorHelper class
		self.lstmed = DetectorHelper(X.shape[1], self.hidden_size,
									 self.n_layers, self.use_bias, self.dropout)
		# move the model to the device specified by the user
		self.to_device(self.lstmed)
		# create an Adam optimizer
		optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)
		# set the model to training mode
		# In PyTorch, the training mode of a model refers to whether the model's parameters are being updated
		# during the forward pass or not. When a model is in training mode, its parameters are being updated
		# based on the gradients computed during the backward pass. When a model is in evaluation mode,
		# its parameters are not updated and certain layers (e.g. dropout layers) may behave differently.
		self.lstmed.train()
		# trange is a wrapper around the range function to provide a smart progress meter
		# when looping over an iterable. It can also be used as a context manager
		for epoch in trange(self.num_epochs):
			# log the epoch number
			logging.debug(f'Epoch {epoch + 1}/{self.num_epochs}.')
			# iterate over the training set
			for ts_batch in train_loader:
				# move the batch to the device specified by the user
				output = self.lstmed(self.to_var(ts_batch))
				# compute the loss between the model's predictions and the actual values using the MSE loss function
				loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
				# clear the gradients
				self.lstmed.zero_grad()
				# compute the gradients
				loss.backward()
				# update the model's parameters
				optimizer.step()
		# set the model to evaluation mode
		# In PyTorch, the evaluation mode of a model refers to a setting in which the model's parameters
		# are not being updated during the forward pass. In other words, when a model is in evaluation mode,
		# the gradients are not computed during the backward pass and the model's parameters are not updated.
		self.lstmed.eval()
		error_vectors = []
		# iterate over the "train Gaussian" set
		for ts_batch in train_gaussian_loader:
			# forward pass
			output = self.lstmed(self.to_var(ts_batch))
			# compute the error between the model's predictions and the actual values
			error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
			# store the error vectors
			error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())
		# compute the mean and covariance of the error vectors
		self.mean = np.mean(error_vectors, axis=0)
		self.cov = np.cov(error_vectors, rowvar=False)
	
	def to_var(self, t, **kwargs):
		'''
		The to_var function converts a tensor to a variable.
		The purpose of this function is to convert the input tensor to a PyTorch Variable,
		which is a wrapper around a tensor that allows the tensor to be used as an input to a computation
		and to store the gradient of the computation with respect to the tensor.

		:param t: tensor to convert
		:param kwargs: keyword arguments
		:return: variable
		'''
		# send the tensor to the device
		t = t.to(self.device)
		# convert the tensor to a variable
		return Variable(t, **kwargs)
	
	def predict(self, X: pd.DataFrame,y_test = None):
		'''
		A multivariate normal distribution is created using the mean and covariance attributes self.mean and self.cov, respectively.
		The model is then applied to each batch of sequences in the data loader, and the error between the model's
		output and the actual sequences is computed. The negative log probability of the error under the
		multivariate normal distribution is also computed and added to a list of scores. If the self.details
		attribute is True, the model's output and the error are also appended to lists.
		Finally, the scores are averaged over each self.sequence_length consecutive entries and returned,
		along with the self.prediction_details dictionary if self.details is True. The self.prediction_details
		dictionary is updated with the mean of the model's output and the mean of the errors, both averaged over
		each self.sequence_length consecutive entries.

		:param X: pandas DataFrame to predict
		:return: numpy array of scores and dictionary of prediction details
		'''
		data = X.values
		sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
		data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)
		# set the model to evaluation mode
		self.lstmed.eval()
		# create a multivariate normal distribution
		mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
		scores = []
		outputs = []
		errors = []
		# iterate over the data loader
		for idx, ts in enumerate(data_loader):
			# forward pass
			output = self.lstmed(self.to_var(ts))
			# compute the error between the model's predictions and the actual values using the L1 loss function
			# The error is calculated using the L1 loss function because it is a commonly used measure of
			# absolute error between two tensors. The L1 loss function is defined as the sum of the absolute
			# differences between the elements of the two tensors. It is often used because it is more robust
			# to outliers than the mean squared error (MSE) loss, which is another commonly used loss function.
			error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
			# compute the negative log probability of the error under the multivariate normal distribution
			# The scores are calculated using the negative log probability of the error under a multivariate
			# normal distribution because the model is assumed to have a Gaussian distribution of errors.
			# The negative log probability is used as a measure of the likelihood of the error under the
			# assumed distribution. This likelihood can then be used to determine whether a particular sample
			# is an outlier or not, as samples with low likelihood are less likely to have been generated by
			# the model and are more likely to be anomalies.
			score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
			scores.append(score.reshape(ts.size(0), self.sequence_length))
			outputs.append(output.cpu().data.numpy())
			errors.append(error.cpu().data.numpy())
		
		# concatenate the scores
		scores = np.concatenate(scores)
		# create a matrix of NaNs with the same shape as the data
		scores_matrix = np.full((self.sequence_length, data.shape[0]), np.nan)
		# iterate over the scores
		for i, score in enumerate(scores):
			# fill the matrix with the scores
			scores_matrix[i % self.sequence_length, i:i + self.sequence_length] = score
		# average the scores over each self.sequence_length consecutive entries
		scores = np.nanmean(scores_matrix, axis=0)
		
		# concatenate the outputs
		outputs = np.concatenate(outputs)
		# create a matrix of NaNs with the same shape as the data
		scores_matrix = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
		# iterate over the outputs
		for i, output in enumerate(outputs):
			# fill the matrix with the outputs
			scores_matrix[i % self.sequence_length, i:i + self.sequence_length, :] = output
		# average the outputs over each self.sequence_length consecutive entries
		self.prediction_details.update({'reconstructions_mean': np.nanmean(scores_matrix, axis=0).T})
		# concatenate the errors
		errors = np.concatenate(errors)
		# create a matrix of NaNs with the same shape as the data
		scores_matrix = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
		# iterate over the errors
		for i, error in enumerate(errors):
			# fill the matrix with the errors
			scores_matrix[i % self.sequence_length, i:i + self.sequence_length, :] = error
		# average the errors over each self.sequence_length consecutive entries
		self.prediction_details.update({'errors_mean': np.nanmean(scores_matrix, axis=0).T})
		# return the scores and the prediction details
		if y_test is not None:
			optimal = self._find_optimal_threshold(y_test, scores)
			predictions = self._convert_scores_to_label(scores, optimal)
			return predictions, self.prediction_details
		return scores, self.prediction_details
	
	def _find_optimal_threshold(self, y_test, y_pred):
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
	
	def _convert_scores_to_label(self, array, threshold):
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


class DetectorHelper(nn.Module):
	def __init__(self, n_features: int, hidden_size: int,
				 n_layers: int, use_bias: bool, dropout: float):
		'''
		The DetectorHelper class is a PyTorch module that contains the LSTMED model.
		The LSTM model has three main components: an encoder, a decoder, and a linear layer.
		The encoder and decoder are both LSTM layers, while the linear layer maps the hidden state of the LSTM
		to the output.
		:param n_features: number of features in the input
		:param hidden_size: number of hidden units in the LSTM
		:param n_layers: is a tuple specifying the number of layers in the encoder and decoder,
		:param use_bias: is a tuple specifying whether to use a bias term in the encoder and decoder,
		:param dropout: is a tuple specifying the dropout probability in the encoder and decoder,
		:param seed: seed for the random number generator
		:param gpu: GPU to use
		'''
		super().__init__()
		# set the device
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		self.n_features = n_features
		self.hidden_size = hidden_size
		
		self.n_layers = n_layers
		self.use_bias = use_bias
		self.dropout = dropout
		# encoder
		self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
							   num_layers=self.n_layers, bias=self.use_bias, dropout=self.dropout)
		# send the encoder to the device
		self.to_device(self.encoder)
		# decoder
		self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
							   num_layers=self.n_layers, bias=self.use_bias, dropout=self.dropout)
		# send the decoder to the device
		self.to_device(self.decoder)
		# linear layer
		self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
		# send the linear layer to the device
		self.to_device(self.hidden2output)
	
	def to_device(self, model):
		'''
		Move the model to the device specified by the user
		:param model: model to move to the device
		'''
		model.to(self.device)
	
	def to_var(self, t, **kwargs):
		'''
		The to_var function converts a tensor to a variable.
		The purpose of this function is to convert the input tensor to a PyTorch Variable,
		which is a wrapper around a tensor that allows the tensor to be used as an input to a computation
		and to store the gradient of the computation with respect to the tensor.

		:param t: tensor to convert
		:param kwargs: keyword arguments
		:return: variable
		'''
		# send the tensor to the device
		t = t.to(self.device)
		# convert the tensor to a variable
		return Variable(t, **kwargs)
	
	def _init_hidden(self, batch_size):
		'''
		The _init_hidden function first creates two tensors of zeros with the appropriate dimensions
		using the torch.Tensor.zero_ method. These tensors are then converted to Variables using the to_var
		method, which moves them to the specified device (CPU or GPU) and wraps them in Variable objects.
		Finally, the function returns a tuple containing the two Variables representing the initial hidden
		state for the encoder and decoder.

		The hidden state of an LSTM is a memory cell that stores information from the past and is used to make
		predictions based on this information. At the beginning of the model's execution, the hidden state should
		be initialized to a tensor of zeros so that it does not contain any information from previous computations.

		By initializing the hidden state to a tensor of zeros, we ensure that the model starts with a blank slate
		and that its predictions are based solely on the current input data, rather than on any information from
		previous computations. The _init_hidden function is used to create the initial hidden state for the
		encoder and decoder of the LSTM model, and this initial hidden state is passed as an argument to the
		forward method when it is called.

		:param batch_size: batch size
		:return: tuple containing the initial hidden state for the encoder and decoder
		'''
		return (self.to_var(torch.Tensor(self.n_layers, batch_size, self.hidden_size).zero_()),
				self.to_var(torch.Tensor(self.n_layers, batch_size, self.hidden_size).zero_()))
	
	def forward(self, ts_batch):
		'''
		The first step in the forward pass is to initialize the hidden state of the encoder using the
		_init_hidden method. The encoder is then applied to the input time series data ts_batch, and the
		output and final hidden state are returned. The final hidden state of the encoder is then used to
		initialize the hidden state of the decoder.

		Next, the output tensor is initialized to a tensor of zeros with the same size as ts_batch. This tensor
		will be used to store the output of the model. The model then iterates over the time steps of ts_batch
		in reverse order, starting from the final time step. At each time step, the hidden2output linear layer
		is applied to the current hidden state of the decoder to produce the output for that time step, which
		is stored in the output tensor.

		If the model is in training mode, the decoder is applied to the input time series data at the current
		time step. If the model is in evaluation mode, the decoder is applied to the output produced by the
		model at the current time step. In either case, the output and final hidden state of the decoder are
		returned and the hidden state of the decoder is updated.

		Why the sequences are processed in reverse order:
			1. Processing the time steps in reverse order allows the model to make predictions for future time steps
			   based on the input data up to the current time step. This is because the hidden state of the LSTM,
			   which stores information from the past, is updated at each time step based on the input data and the
			   previous hidden state. By starting at the final time step and working backwards, the hidden state
			   will contain information about all of the input data up to the current time step, which can be used
			   to make predictions for future time steps.
			2. Processing the time steps in reverse order allows the model to take advantage of the temporal
			   dependencies in the data. LSTM models are able to capture long-term dependencies in time series data,
			   and processing the time steps in reverse order may allow the model to more easily capture these
			   dependencies.

		:param ts_batch: batch of time series data
		:param return_latent: whether to return the latent representation
		:return: output of the model
		'''
		# get the batch size
		batch_size = ts_batch.shape[0]
		# initialize the hidden state of the encoder
		enc_hidden = self._init_hidden(batch_size)
		# apply the encoder to the input time series data
		_, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)
		# initialize the hidden state of the decoder
		dec_hidden = enc_hidden
		# initialize the output tensor
		output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
		# This means that the model starts at the final time step of the input data and works its way
		# backwards to the initial time step.
		
		# iterate over the time steps of the input time series data
		for i in reversed(range(ts_batch.shape[1])):
			# apply the linear layer to the current hidden state of the decoder
			output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])
			if self.training:
				# apply the decoder to the input time series data at the current time step
				_, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
			else:
				# apply the decoder to the output produced by the model at the current time step
				_, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
		
		# return the output of the model
		return output