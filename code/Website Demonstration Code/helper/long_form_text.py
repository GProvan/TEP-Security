import streamlit as st
def training_text():
	st.markdown(
		'''
		# Anomaly Detection 🤖
		Here we will train a model to detect anomalies in the Tennessee Eastman process, and then use the model to detect anomalies in the test set.
		''')
	# button to make predictions
	st.markdown(
		"""
		## Model Parameters Explained

		- `Batch Size` The number of samples to use in each batch
		- `Number of Epochs` The number of times to iterate over the entire dataset
		- `Learning Rate` The learning rate for the model
		- `Hidden Size` The number of nodes in the hidden layer
		- `Number of Layers` The number of layers in the model
		- `Dropout` The probability of randomly dropping out nodes during training to prevent overfitting
		- `Sequence Length` The number of time steps to use in each sequence
		- `Use Bias` Whether to include bias in the LSTM computations
	""")
	
def correlation_text():
	st.markdown(
		'''
		# Correlation Subgraph

		### What is a correlation subgraph?
		Here we describe how a correlation graph can be generated to localise the cause of an anomaly in a cyber physical system.
		To create this graph, we use the Spearman rank correlation coefficient, which is a non-parametric measure of correlation based
		on the ranks of the data. This coefficient is robust to non-normality and can handle both ordinal and continuous variables,
		making it suitable for use in a cyber physical system where data may not always be normally distributed.

		### How does it work?
		The correlation coefficient is calculated between all pairs of features in the system, and if the absolute value of their correlation coefficient
		is above a user-defined threshold, we create an edge between them in the correlation graph. This threshold is implemented
		to ensure disconnected subgraphs are generated. The resulting graph will consist of a set of disconnected
		subgraphs, where each subgraph is a group of features that are highly correlated with each other.
		These subgraphs can then be used to localise the cause of the anomaly.

		👈 **Set the minimum correlation threshold in the sidebar and click the button to generate the correlation subgraph.**


		'''
	)
	
def pca_text():
	st.markdown(
		'''
		# PCA Localisation 📌

		### What is PCA Localisation?
		Principal Component Analysis (PCA) is a statistical technique that transforms a dataset into a new
		coordinate system where the data is represented by a set of uncorrelated variables, known as principal
		components. The principal components are ordered in terms of their contribution to the variance in the data.

		### How does it work?
		PCA can be used for anomaly detection in cyber-physical systems by analyzing the reconstruction error
		from the test predictions of our proposed. The reconstruction error can be used to identify anomalous
		samples, and performing PCA on the reconstruction error allows us to identify the features that contributed
		the most to the anomaly. This information can be used to diagnose the cause of the anomaly and
		potentially take corrective action.

		👈 **Set the number of components you want returned and  click the button to generate the localisation**


		'''
	)
	
def threshold_text():
	st.markdown(
		'''
		# Threshold Localisation 📌

		### What is Threshold Localisation?
		This section discusses how to use reconstruction error for each feature in time series data from an
		industrial control system to localize the cause of an anomaly. The process involves comparing the
		reconstruction error for each feature in the anomalous data to the maximum reconstruction error for that
		feature on the normal operation data.

		### How does it work?
		By setting a threshold value for each feature and identifying the features with the most time steps above
		the threshold, we can determine the features that are likely causing or closely related to the anomaly.


		👈 **Set the number of components you want returned and  click the button to generate the localisation**


		'''
	)
	
def explainer_text():
	st.markdown(
		'''
		# Anomaly Explanation 🚨

		### What is Anomaly Explanation?
		The proposed approach uses an unsupervised model to generate predictions, which are then used to train a
		supervised explanation model. The explanation model leverages Gradient Boosting Machines (GBMs) to
		generate interpretable and actionable rules in the form of if-then statements.
		GBMs are a popular machine learning model that iteratively adds decision trees to the model to
		improve prediction accuracy.

		### How does it work?
		The GBM model can be fitted to the training data by minimizing the loss function using gradient descent.
		GBMs can be interpreted by examining the importance of each feature in the model, which is calculated by
		measuring how much the model's accuracy decreases when a feature is randomly shuffled. GBMs also
		provide information about the contribution of each feature to each decision tree in the model.

		The generated if-then rules are expressed in terms of the input features and the predicted labels and
		can be transformed into actionable insights. The rules can be visualized easily due to the tree-based
		structure of the GBM, which allows the decision trees in the model to be displayed graphically.

		### What are the benefits?
		The importance of human-readable and actionable insights is that they enable workers in the CPS to quickly identify and correct anomalies, reducing the impact of the anomaly on the system.


		👈 **Pick the data point to describe and click the button to generate the explanation**


		'''
	)
	
	