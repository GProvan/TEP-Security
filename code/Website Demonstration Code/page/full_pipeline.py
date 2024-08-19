def pipeline():
	import streamlit as st
	from cadlae.detector import AnomalyDetector
	from helper.st_utils import data_preprocess
	from cadlae.correlationSubgraph import CorrelationSubgraph
	from cadlae.localisationPCA import PCALocalization
	from cadlae.localisationFeatureWise import FeatureWiseLocalisation
	from cadlae.explainer import ActionExplainer
	import torch
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.metrics import roc_curve,roc_auc_score, confusion_matrix, classification_report
	from helper.st_utils import plot_roc_curve, metric_table, make_confusion_matrix
	import streamlit.components.v1 as components
	st.markdown(
		'''
		# CADLAE Full Pipeline Demo
		
		Here we will demonstrate the full pipeline of CADLAE. We will first train the model on a dataset of normal data. Then we will test the model on a dataset of anomalous data. Finally, we will test the model on a dataset of normal data.
		
		
		''')
	st.sidebar.header('Select Dataset 📂')
	# select box for dataset
	dataset_dict = {'IDV(4)': './data/test_data_idv4.csv'}
	dataset = st.sidebar.selectbox('Select the dataset', ['IDV(4)'])

	from helper.user_parameters import training_parameters
	pretrained, batch_size, epochs, learning_rate, hidden_size, num_layers, sequence_length, dropout, use_bias = training_parameters()
	# true false

	
	st.sidebar.header('Model Localisation')
	st.sidebar.subheader('Correlation Graph')
	corr_graph = st.sidebar.checkbox('Show Correlation Graph', value=True)
	corr = st.sidebar.slider('Select the minimum correlation', 0.1, 0.9, 0.6, 0.01)
	st.sidebar.subheader('Localisation')
	localisation_method = st.sidebar.radio(
		"Choose localisation method",
		('PCA', 'Thresholding'))
	
	num_variables = st.sidebar.slider("Top K most likely variables", 1, 52, 5)
	
	st.sidebar.header('Explanation')
	index = st.sidebar.number_input("Enter index of data point to explain", min_value=0, max_value=10_000,
									value=550, step=1)
	
	if st.button('Run CADLAE'):
	
		
		# Data Processing
	
		X_train, y_train, X_test, y_test, col_names, scaler = data_preprocess("./data/train_data.csv", dataset_dict[dataset])
		
		# Training -> yes we want to train the model
		if pretrained == False:
			model = AnomalyDetector(batch_size=batch_size, num_epochs=epochs, lr=learning_rate,
									hidden_size=hidden_size, n_layers=num_layers, dropout=dropout,
									sequence_length=sequence_length, use_bias=use_bias,
									train_gaussian_percentage=0.25)
			
			with st.spinner('Training Model...'):
				model.fit(X_train)
		
		else:
			try:
				model = torch.load("./model/model_demo.pth")
			except:
				model = torch.load("./model/backup.pth")
			
			
		# Testing -> yes we want to test the model
		with st.spinner('Testing Model...'):
			y_pred, details = model.predict(X_test)
		
		st.header('Results')
		with st.spinner("Predicting on test data..."):
			y_pred, details = model.predict(X_test, y_test)
		st.subheader('Metrics 📊')
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
		# Localisation
		st.header('Localisation')
		if corr_graph:
			with st.spinner('Generating Correlation Graph...'):
				st.subheader('Correlation Subgraph with threshold = ' + str(corr))
				subgraph = CorrelationSubgraph(X_train, corr)
				subgraph.plot_corr_graph_st()
				
				st.subheader('Subgraphs Generated')
				for key, value in subgraph.generate_subgraph_dict().items():
					st.write(str(key) + ': ' + ', '.join(value))
			
		# localisation
		with st.spinner('Localising Anomalies...'):
			if localisation_method == 'PCA':
				
				pca_localization = PCALocalization(3)
				pca_localization.fit(details["errors_mean"])
				# data_pca = pca_localization.transform(details["errors_mean"])
				result = pca_localization.localise(num_variables, col_names)
				
			st.subheader("Top {} most likely causes of anomaly".format(num_variables))
			for key, value in result.items():
				num = str(key) + ". "
				st.write(num, value)
				
			st.subheader("PCA Plot")
			with st.spinner("Plotting PCA..."):
				pca_localization = PCALocalization(3)
				pca_localization.pca_3D_st(details["errors_mean"], y_test)
					
			if localisation_method == 'Thresholding':
				t_scores, d_train = model.predict(X_train)
				train_scores, details_train = t_scores.copy(), d_train.copy()
				test_preds, details_test = model.predict(X_test)
			
		
				ftwise = FeatureWiseLocalisation(y_test, test_preds,col_names, details_train, details_test)
				rank, y_predictions = ftwise.run()
			
				st.subheader("Top {} most likely causes of anomaly".format(num_variables))
		
			
				lst_sorted = sorted(rank, key=lambda x: x[1][0], reverse=True)[:num_variables]  # sort by number of threshold violations
				for i, (feat, (violations, percentage)) in enumerate(lst_sorted):
					st.write(f"{i + 1}. {feat} with {violations} threshold violations ({percentage:.2f}%)")
			
		# Pretrained -> yes we want to use the pretrained model
		st.header('Explanation and Action')
		st.warning(
			'Due to compatibility issues, the following charts are not dynamically sized, as a result, they may not be displayed correctly. We apologise for any inconvenience caused.')
		with st.spinner('Explainer is learning, Please Wait...'):
			mod = ActionExplainer()
			mod.fit(X_test, y_test, max_depth=4)
			mod.learn_data()
			
			def st_dtree(plot, height=None):
				# dtree_html = f"<body>{plot.view().svg()}</body>"
				dtree_html = f"<div style='text-align:center'><body>{plot.view().svg()}</body></div>"
				
				components.html(dtree_html, height=height, width=750)
			
			st.subheader('Global Explanation')
			st_dtree(mod.model, 550)
			
			def st_dtree_local(plot, height=None, width=None):
				dtree_html = f"<div style='text-align:center'><body>{plot.view(x=X_test.iloc[index], show_just_path=True).svg()}</body></div>"
				
				# dtree_html = f"<body>{plot.view(x=X_test.iloc[index], show_just_path=True).svg()}</body>"
				
				components.html(dtree_html, height=height, width=width)
			
			st.subheader('Local Explanation for data point {}'.format(index))
			st_dtree_local(mod.model, 600, 800)
			prediction_proba = mod.clf.predict_proba(X_test.iloc[index].values.reshape(1, -1))
			if prediction_proba[0][0] > prediction_proba[0][1]:
				mode = "Normal Activity"
				prob = round(prediction_proba[0][0] * 100, 2)
			else:
				mode = "Fault"
				prob = round(prediction_proba[0][1] * 100, 2)
			
			st.markdown("""
			<style>
			.big-font {
			    font-size:18px !important;
			}
			</style>
			""", unsafe_allow_html=True)
			
			st.subheader('Report for data point: {} 🔍'.format(index))
			
			st.markdown(
				'<p class="big-font">Model prediction for data point {}: <strong>{}</strong></p>'.format(index, mode),
				unsafe_allow_html=True)
			st.markdown(
				'<p class="big-font">Probability of data point {} being in class {}: <strong>{}%.</strong></p>'.format(
					index, mode, prob), unsafe_allow_html=True)
			st.subheader('Suggested Action 💡')
			actions = mod.action(index)
			for action, description in actions.items():
				st.write('<p class="big-font"><strong>{}: </strong>{}</p>'.format(action, description),
						 unsafe_allow_html=True)
			


	