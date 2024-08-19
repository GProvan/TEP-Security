def generate_pca_localisation():
	from cadlae.localisationPCA import PCALocalization
	from cadlae.preprocess import DataProcessor
	from cadlae.detector import AnomalyDetector
	import streamlit as st
	import torch
	
	from helper.st_utils import data_preprocess
	# Data Preprocessing
	train = "data/train_data.csv"
	test = "data/test_data_idv4.csv"
	X_train, y_train, X_test, y_test, col_names, scaler = data_preprocess(train, test)
	num_variables = st.sidebar.slider("Top K most likely variables", 1, len(col_names), 5)
	
	
	from helper.long_form_text import pca_text
	pca_text()
	
	if st.button("PCA Localisation"):
	
		
		
		with st.spinner('Model is Training, Please Wait...'):
			try:
				model = torch.load("./model/model_demo.pth")
			except:
				model = torch.load("./model/backup.pth")
		
			
		with st.spinner('Making Predictions, Please Wait...'):
			y_pred, details = model.predict(X_test, y_test)
		
		with st.spinner("Using Predictions to Localise Anomalies..."):
			pca_localization = PCALocalization(3)
			pca_localization.fit(details["errors_mean"])
			result = pca_localization.localise(num_variables, col_names)
		
		st.subheader("Top {} most likely causes of anomaly".format(num_variables))
		for key,value in result.items():
			num = str(key) + ". "
			st.write(num,value)
			
		st.subheader("PCA Plot")
		with st.spinner("Plotting PCA..."):
			pca_localization = PCALocalization(3)
			pca_localization.pca_3D_st(details["errors_mean"], y_test)
		
		
		
			
		
			
	
			
			
			
			
			
			
			
		