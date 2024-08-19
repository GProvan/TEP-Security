def generate_corr_subgraph():
	from cadlae.preprocess import DataProcessor
	from cadlae.correlationSubgraph import CorrelationSubgraph
	from cadlae.localisationSubgraph import LocaliseSubgraph
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.metrics import roc_curve,roc_auc_score
	import seaborn as sns
	import networkx as nx
	
	train_link = "data/train_data.csv"
	test_link = "data/test_data_idv4.csv"
	processor = DataProcessor(train_link, test_link, "Fault", "Unnamed: 0")
	X_train = processor.X_train
	import streamlit as st
	
	
	
	
	from helper.long_form_text import correlation_text
	
	correlation_text()
	corr = st.sidebar.slider('Select the minimum correlation', 0.1, 0.9, 0.6, 0.01)
	if st.button('Generate Correlation Subgraph! 🚀'):
		st.subheader('Correlation Subgraph with threshold = ' + str(corr))
		subgraph = CorrelationSubgraph(X_train, corr)
		subgraph.plot_corr_graph_st()
		
		st.subheader('Subgraphs Generated')
		for key, value in subgraph.generate_subgraph_dict().items():
			st.write(str(key) + ': ' + ', '.join(value))
			
			
			
		
			
		
		
		