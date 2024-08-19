import pandas as pd
import numpy as np
import warnings
from causalnex.structure import StructureModel
import networkx as nx
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.network import BayesianNetwork


class BayesianInference:
	
	def __init__(self, graph_structure):
		self.sm = StructureModel()
		self.sm = nx.drawing.nx_pydot.read_dot(graph_structure)
		self.sm.remove_node("\\n")
	
	def _remove_brackets(self, df):
		old_to_new_names = {}
		for col in df.columns:
			new_col = col.replace('(', '').replace(')', '')
			old_to_new_names[col] = new_col
		df = df.rename(columns=old_to_new_names)
		return df, old_to_new_names
	
	def _quantile_discretize(self, df):
		cols = list(df.columns)
		for col in cols:
			q = [0, 0.05, 0.2, 0.8, 0.95, 1]
			bins = df[col].quantile(q)
			labels = ["Very Low", "Low", "Normal", "High", "Very High"]
			df[col] = pd.cut(df[col], bins=bins, labels=list(bins)[:-1], include_lowest=True)
		
		return df
	
	def define_discretize_variable_states(self, df):
		cols = list(df.columns)
		labels = ["Very Low", "Low", "Normal", "High", "Very High"]
		for col in cols:
			cutoff = sorted(list(pd.unique(df[col])))
			cutoff_dict = {cutoff[i]: labels[i] for i in range(len(cutoff))}
			# d = {val: "G" + str(i+1) for i, val in enumerate(list(pd.unique(df[col])))}
			df[col] = df[col].map(cutoff_dict)
		
		return df
	
	def fit_state(self, data):
		df, dict_name = self._remove_brackets(data)
		df = self._quantile_discretize(df)
		df = self._define_discretize_variable_states(df)
		self.sm = nx.relabel_nodes(self.sm, dict_name)
		self.bn = BayesianNetwork(self.sm)
		self.bn = bn.fit_node_states(df)
	
	def fit_cpt(self, df):
		self.bn = self.bn.fit_cpds(df, method="BayesianEstimator", bayes_prior="K2")
	
	def plot_network(self):
		viz = plot_structure(self.sm, graph_attributes={"scale": "1.5"},
							 all_node_attributes=NODE_STYLE.NORMAL,
							 all_edge_attributes=EDGE_STYLE.WEAK)
		filename = "./structure_model.png"
		viz.draw(filename, prog="twopi")
		Image(filename)