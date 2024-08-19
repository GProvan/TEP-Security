from cadlae.detector import AnomalyDetector
from cadlae.preprocess import DataProcessor
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def data_preprocess(train,test):
	processor = DataProcessor(train, test, "Fault", "Unnamed: 0")
	X_train = processor.X_train
	y_train = processor.y_train
	X_test = processor.X_test
	y_test = processor.y_test
	scaler = processor.scaler_function
	col_names = processor.col_names
	return X_train, y_train, X_test, y_test,col_names, scaler

# X_train, y_train, X_test, y_test,col_names, scaler = data_preprocess(train,test)


def fit_and_predict(model, X_train, X_test, y_test):
	model.fit(X_train)
	y_pred, details = model.predict(X_test, y_test)
	return model, y_pred, details

# model, y_pred, details = fit_and_predict(model, X_train, X_test, y_test)


def metric_table(y_test, y_pred):
	cm = confusion_matrix(y_test, y_pred)
	TP = cm[0, 0]
	TN = cm[1, 1]
	FP = cm[0, 1]
	FN = cm[1, 0]
	accuracy = accuracy_score(y_test, y_pred)
	precision = TP / float(TP + FP)
	recall = TP / float(TP + FN)
	f1 = 2 * (precision * recall) / (precision + recall)
	roc = roc_auc_score(y_test, y_pred)
	
	# multiple *100, round to 2 decimal places, and add % sign
	accuracy = round(accuracy * 100, 2)
	precision = round(precision * 100, 2)
	recall = round(recall * 100, 2)
	f1 = round(f1 * 100, 2)
	roc = round(roc * 100, 2)
	
	accuracy = str(accuracy) + '%'
	precision = str(precision) + '%'
	recall = str(recall) + '%'
	f1 = str(f1) + '%'
	roc = str(roc) + '%'
	
	
	
	return accuracy, precision, recall, f1, roc

#accuracy, precision, recall, f1, roc = metric_table(y_test, y_pred)

def make_confusion_matrix(y_true, y_prediction, c_map="viridis"):
	sns.set(font_scale=0.8)
	fig, ax = plt.subplots()
	ax.set_title('CADLAE Confusion Matrix')
	cm = confusion_matrix(y_true, y_prediction)
	
	cm_matrix = pd.DataFrame(data=cm, columns=['Normal', 'Attack'],
							 index=['Normal', 'Attack'])
	
	sns.heatmap(cm_matrix, annot=True, fmt='.0f', cmap=c_map, linewidths=1, linecolor='black', clip_on=False)
	st.pyplot(fig)


def plot_roc_curve(fpr, tpr):
	fig, ax = plt.subplots()
	ax.set_facecolor('white')  # set background color to white
	ax.spines['bottom'].set_color('black')  # set color of x-axis to black
	ax.spines['left'].set_color('black')
	plt.plot(fpr, tpr, color='orange', label='ROC')
	plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	st.pyplot(fig)
	
def get_attacks(y_test, outlier=1, normal=0, breaks=[]):
	'''
	Get indices of anomalies
	:param y_test: predictions from semi supervised model
	:param outlier: label for anomalies
	:param normal: label for normal data points
	:param breaks: indices of breaks in data
	:return:
	'''
	events = dict()
	label_prev = normal
	event = 0  # corresponds to no event
	event_start = 0
	for tim, label in enumerate(y_test):
		if label == outlier:
			if label_prev == normal:
				event += 1
				event_start = tim
			elif tim in breaks:
				# A break point was hit, end current event and start new one
				event_end = tim - 1
				events[event] = (event_start, event_end)
				event += 1
				event_start = tim
		
		else:
			# event_by_time_true[tim] = 0
			if label_prev == outlier:
				event_end = tim - 1
				events[event] = (event_start, event_end)
		label_prev = label
	
	if label_prev == outlier:
		event_end = tim - 1
		events[event] = (event_start, event_end)
	return events

def get_attack_idx_list(dictionary):
	'''
	Get list of indices of anomalies
	:param dictionary: dictionary of anomalies
	:return: Dictionary of anomalies, value is changed from (start, end) to list of indices
	'''
	for key, value in dictionary.items():
		if isinstance(value, tuple):
			dictionary[key] = list(range(value[0], value[1] + 1))
	return dictionary


def plot_anomalies(df, column, anomalies, scaler=None):
	'''
	Plot anomalies
	:param df: dataframe to plot
	:param column: column to plot
	:param anomalies: dictionary of anomalies -> pass through dictionary - list pipeline to get dictionary with indx of anomalies
	:param reverse_scaler: object used to scale the data -> reverses to original scale in plot of passed
	'''
	fig, ax = plt.subplots()
	ax.set_facecolor('white')  # set background color to white
	ax.spines['bottom'].set_color('black')  # set color of x-axis to black
	ax.spines['left'].set_color('black')
	if scaler is not None:
		df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
	title = "Plot of {}".format(column)
	ax.plot(df[column])
	ax.set_title(title)
	for key, value in anomalies.items():
		ax.plot(value, df[column][value], 'ro', markersize=4, color='red')
	st.pyplot(fig)


def plot_reconstructions(details, X, column):
	fig, ax = plt.subplots()
	ax.set_facecolor('white')  # set background color to white
	ax.spines['bottom'].set_color('black')  # set color of x-axis to black
	ax.spines['left'].set_color('black')
	ax.plot(X[column], label='original series')
	col_idx = X.columns.get_loc(column)
	ax.plot(details['errors_mean'][col_idx], label='reconstructed error mean', color='red')
	ax.set_title('Reconstructions of column {}'.format(column))
	st.pyplot(fig)