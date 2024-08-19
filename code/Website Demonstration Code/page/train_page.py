
def prediction():
	# import libraries
	import streamlit as st
	import pandas as pd
	from cadlae.detector import AnomalyDetector
	from sklearn.metrics import roc_curve, classification_report
	
	from helper.long_form_text import training_text
	from helper.user_parameters import training_parameters
	from helper.st_utils import data_preprocess, fit_and_predict, metric_table, make_confusion_matrix, plot_roc_curve
	from helper.st_utils import get_attacks, get_attack_idx_list
	# function for intro text
	training_text()
	
	# function for user parameters
	pretrained, batch_size, epochs, learning_rate, hidden_size, num_layers, sequence_length, dropout, use_bias = training_parameters()

	
	
	if st.button('Train the Model! 🚀'):
		# instantiate model
		model = AnomalyDetector(batch_size=batch_size, num_epochs=epochs, lr=learning_rate,
								hidden_size=hidden_size, n_layers=num_layers, dropout=dropout,
								sequence_length=sequence_length, use_bias=use_bias,
								train_gaussian_percentage=0.25)
		
		
		# Data Preprocessing
		train = "data/train_data.csv"
		test = "data/test_data_idv4.csv"
		X_train, y_train, X_test, y_test, col_names, scaler = data_preprocess(train, test)
		
	
		# Train Model and Predict
		with st.spinner('Model is Training, Please Wait...'):
			model, y_pred, details = fit_and_predict(model, X_train, X_test, y_test)
		
		st.header('Model Training Complete! 🎉')
		st.markdown('''
        The model has been trained in an unsupervised manner, and has learned the normal behaviour of the process.
        We have fit the model on unseen data using the parameters you have selected above. The results are shown below.
        ''')
		st.header('Model Performance 📈')
		st.subheader('Model Metrics 📊')
		
		accuracy, precision, recall, f1, roc = metric_table(y_test, y_pred)
		# add metrics to dataframe, with columns Metric and Value
		metrics = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'],
								'Result': [accuracy, precision, recall, f1, roc]})
		st.table(metrics)
		
		st.subheader('Classification Report 📝')
		report = classification_report(y_test, y_pred, output_dict=True)
		report = pd.DataFrame(report).transpose()
		st.table(report)
		
		
		st.subheader('Confusion Matrix')
		make_confusion_matrix(y_test, y_pred, c_map="Blues")
		
		st.subheader('ROC-AUC Curve')
		fpr, tpr, thresholds = roc_curve(y_test, y_pred)
		plot_roc_curve(fpr, tpr)
		
	
		
		dict_attacks = get_attacks(y_pred, outlier=1, normal=0, breaks=[])
		attacks = get_attack_idx_list(dict_attacks)
		st.subheader('Predicted Anomalies')
		from helper.st_utils import plot_anomalies
		plot_anomalies(X_test, "XMV(10)", attacks, scaler)
		
		st.subheader('LSTM Reconstruction Error')
		from helper.st_utils import plot_reconstructions
		plot_reconstructions(details, X_test, "XMV(10)")
