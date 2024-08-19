def generate_threshold_localisation():
	from cadlae.localisationFeatureWise import FeatureWiseLocalisation
	import streamlit as st
	import torch
	
	from helper.long_form_text import threshold_text
	threshold_text()
	
	from helper.st_utils import data_preprocess
	# Data Preprocessing
	train = "data/train_data.csv"
	test = "data/test_data_idv4.csv"
	X_train, y_train, X_test, y_test, col_names, scaler = data_preprocess(train, test)

	num_variables = st.sidebar.slider("Top K most likely variables", 1, len(col_names), 3)

	
	

	
	
	if st.button("Feature Wise Localisation"):
		with st.spinner('Model is Training, Please Wait...'):
			try:
				model = torch.load("./model/model_demo.pth")
			except:
				
				model = torch.load("./model/backup.pth")
		
			

		
		with st.spinner('Making Predictions, Please Wait...'):
			t_scores, d_train = model.predict(X_train)
			train_scores, details_train = t_scores.copy(), d_train.copy()
			test_preds, details_test = model.predict(X_test)
		
		with st.spinner("Using Predictions to Localise Anomalies..."):
			ftwise = FeatureWiseLocalisation(y_test, test_preds, col_names, details_train, details_test)
			rank, y_predictions = ftwise.run()
		
			
		st.subheader("Top {} most likely causes of anomaly".format(num_variables))
	
		
		lst_sorted = sorted(rank, key=lambda x: x[1][0], reverse=True)[:num_variables]  # sort by number of threshold violations
		for i, (feat, (violations, percentage)) in enumerate(lst_sorted):
			st.write(f"{i + 1}. {feat} with {violations} threshold violations ({percentage:.2f}%)")
	