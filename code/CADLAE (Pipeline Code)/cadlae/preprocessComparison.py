import pandas as pd
import numpy as np
from sklearn import preprocessing


def preprocess_datasets(dataset_name, train_link, percent_training=None, test_link=None):
	# if test link is none, its a singl;e dataset and thus needs to be split accordingly
	"""
	This function preprocesses datasets for machine learning. It can handle multiple datasets including "gecco", "swat",
	and "adbench". It returns the training and test datasets as well as the column names.

	Parameters:
	dataset_name (str): The name of the dataset to be preprocessed.
	train_link (str): The filepath for the training dataset.
	percent_training (float): If the dataset is "adbench", this parameter specifies the percentage of data to use for training.
	test_link (str): The filepath for the test dataset. If test_link is None, the function assumes the dataset is "adbench"
	and splits it into training and test datasets.

	Returns:
	X_train (pandas.DataFrame): The training dataset without the target variable.
	y_train (pandas.Series): The target variable for the training dataset.
	X_test (pandas.DataFrame): The test dataset without the target variable.
	y_test (pandas.Series): The target variable for the test dataset.
	col_names (list): The column names for the training and test datasets.

	"""
	
	try:
		if dataset_name.lower() in ["gecco"]:
			pass
	except:
		return "Error, dataset name not recognised"
	
	if test_link is not None:
		try:
			train_data = pd.read_csv(train_link)
			test_data = pd.read_csv(test_link)
		except:
			return "Error: datasets could not be loaded, check the links and try again."
		################ GECCO DATASET ###############
		if dataset_name.lower() == "gecco":
			# Generate Training data
			train_data = train_data.dropna()
			train_data = train_data[train_data.Event != 1]
			train_data[train_data.isna().any(axis=1)]
			train_data['Event'] = train_data['Event'].astype(int)
			y_train = train_data['Event']
			X_train = train_data.drop(columns=['Time', 'Unnamed: 0', 'Event'], axis=1)
			col_names = list(X_train.columns)
			
			# Generate Test Data
			test_data = pd.read_csv(test_link)
			test_data = test_data.dropna()
			test_data[test_data.isna().any(axis=1)]
			test_data['Event'] = test_data['Event'].astype(int)
			y_test = test_data['Event']
			X_test = test_data.drop(columns=['Time', 'Unnamed: 0', 'Event'], axis=1)
		
		elif dataset_name.lower() == "swat":
			y_train = train_data["Normal/Attack"]
			X_train = train_data.drop([" Timestamp", "Normal/Attack"], axis=1)
			y_test = test_data["Normal/Attack"]
			X_test = test_data.drop([" Timestamp", "Normal/Attack"], axis=1)
	
	elif test_link is None:
		if dataset_name.lower() == "adbench":
			print("here")
			df = np.load(train_link, allow_pickle=True)
			X, y = df['X'], df['y']
			X = pd.DataFrame(X)
			y = pd.DataFrame(y, columns=["Fault"])
			data = pd.concat([X, y], axis=1)
			training_index = int(round(len(data) * percent_training, 0))
			X_train = data[data['Fault'] != 1].iloc[:training_index]
			y_train = X_train["Fault"]
			X_train.drop('Fault', axis=1, inplace=True)
			col_names = list(X_train.columns)
			X_test = data[~data.index.isin(X_train.index)]
			y_test = X_test["Fault"]
			X_test.drop('Fault', axis=1, inplace=True)
	
	# Scale Data - should hopefully be consistent across the datasets, make sure all names are defined
	
	scaler = preprocessing.StandardScaler()
	x_scaled_train = scaler.fit_transform(X_train)
	X_train = pd.DataFrame(x_scaled_train, columns=col_names)
	
	x_scaled_test = scaler.transform(X_test)
	X_test = pd.DataFrame(x_scaled_test, columns=col_names)
	return X_train, y_train, X_test, y_test, col_names

# X_train, y_train, X_test, y_test, col_names = preprocess_datasets("gecco", train_link,