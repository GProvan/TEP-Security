import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import streamlit as st
import seaborn as sns
sns.set_style("whitegrid")


class PCALocalization:
	def __init__(self, components=1):
		"""
		A class for localizing the principal components of a dataset using PCA.

		Attributes:
		-----------
		pca : PCA object
			An instance of PCA from the scikit-learn library.

		Methods:
		--------
		__init__(self, components=1):
			Initializes the PCALocalization object with the number of components to extract.

		fit(self, error_means):
			Fits the PCA model to the input data.

		transform(self, data):
			Applies the fitted PCA model to a new dataset and returns the transformed data.

		localise(self, k_features=None, col_names=None):
			Returns the top k_features that contributed the most to the principal components.
		"""
		self.pca = PCA(n_components=components)
	
	def fit(self, error_means):
		'''
		Fits the PCA model to the input data.
		:param error_means: A pandas dataframe containing the error means for each sensor.
		:return: None
		'''
		error = error_means.T
		data = pd.DataFrame(error)
		self.pca.fit(data)
	
	def transform(self, data):
		'''
		Applies the fitted PCA model to a new dataset and returns the transformed data.
		:param data: A pandas dataframe containing the data to be transformed.
		:return: A numpy array containing the transformed data.
		'''
		data_T = data.T
		data_pca = self.pca.transform(data_T)
		return data_pca
	
	def localise(self, k_features=None, col_names=None):
		'''
		Returns the top k_features that contributed the most to the principal components.
		:param k_features: The number of features to return.
		:param col_names: A list of the column names.
		:return: A pandas dataframe containing the top k_features.
		'''
		if k_features is None:
			k_features = 1
		top_k_features = np.argsort(np.abs(self.pca.components_), axis=1)[:, -k_features:]
		result = []
		for i, top_features in enumerate(top_k_features):
			result.append(list(top_features))
		result = result[0][::-1]
		col_dict = {}
		i = 0
		while i < len(result):
			if col_names is None:
				col = result[i]
			else:
				col = col_names[result[i]]
			col_dict[i + 1] = col
			i += 1
		return col_dict
	
	def pca_2D(self, data,y_test, width: int = 10, height: int = 8):
		'''
		Plots the first two principal components of the data.
		:param data: A pandas dataframe containing the data to be transformed.
		:param width: The width of the plot.
		:param height: The height of the plot.
		:return: None
		'''
		pca_localization = PCALocalization(2)
		pca_localization.fit(data)
		data_pca = pca_localization.transform(data)
		fig = plt.figure(figsize=(width, height))
		x = data_pca[:, 0]
		y = data_pca[:, 1]
		x = pd.Series(x)
		y = pd.Series(y)
		df1 = pd.concat([x, y, y_test], axis=1).reset_index()
		cmap = colors.ListedColormap(['red', 'blue'])
		plt.scatter(x, y, c=df1["Fault"], cmap=cmap)
		plt.xlabel("First Principal Component")
		plt.ylabel("Second Principal Component")
		plt.title("PCA Plot Colored by Target Variable")
		plt.show()
	
	def pca_3D(self, data,y_test, width: int = 10, height: int = 8, xlim_left: int = 2,
			   xlim_right: int = 7):
		'''
		Plots the first three principal components of the data.
		:param data: A pandas dataframe containing the data to be transformed.
		:param width: The width of the plot.
		:param height: The height of the plot.
		:param xlim_left: The left limit of the x-axis.
		:param xlim_right: The right limit of the x-axis.
		:return: None
		'''
		# Example usage
		pca_localization = PCALocalization(3)
		pca_localization.fit(data)
		data_pca = pca_localization.transform(data)
		x = data_pca[:, 0]
		y = data_pca[:, 1]
		z = data_pca[:, 2]
		df1 = pd.concat([x, y, y_test], axis=1).reset_index()
		fig = plt.figure(figsize=(width, height))
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlim(left=-xlim_left, right=xlim_right)
		ax.scatter(x, y, z, c=df1["Fault"])
		ax.set_xlabel("First Principal Component")
		ax.set_ylabel("Second Principal Component")
		ax.set_zlabel("Third Principal Component")
		plt.title("PCA Plot Coloured by Target Variable")
		plt.show()
		
	def pca_3D_st(self, data,y_test, width: int = 10, height: int = 8, xlim_left: int = 2,
			   xlim_right: int = 7):
		'''
		Plots the first three principal components of the data.
		:param data: A pandas dataframe containing the data to be transformed.
		:param width: The width of the plot.
		:param height: The height of the plot.
		:param xlim_left: The left limit of the x-axis.
		:param xlim_right: The right limit of the x-axis.
		:return: None
		'''
		# Example usage
		pca_localization = PCALocalization(3)
		pca_localization.fit(data)
		data_pca = pca_localization.transform(data)
		x = data_pca[:, 0]
		y = data_pca[:, 1]
		z = data_pca[:, 2]
		fig, ax = plt.subplots(figsize=(15, 15))
		ax.axis('off')
		plt.xlabel('', fontsize=15)
		plt.ylabel('', fontsize=15)
		plt.ylabel('', fontsize=15)
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlim(left=-xlim_left, right=xlim_right)
		cmap = colors.ListedColormap(['red', 'blue'])
		ax.scatter(x, y, z, c=y_test, cmap=cmap)
		ax.set_xlabel("First Principal Component")
		ax.set_ylabel("Second Principal Component")
		ax.set_zlabel("Third Principal Component")
		st.pyplot(fig)
		