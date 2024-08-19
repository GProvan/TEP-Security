import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import dtreeviz
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dtreeviz import decision_boundaries
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import seaborn as sns
sns.set_style("whitegrid")
class ActionExplainer:
	def __init__(self, variable_description: bool = True):
		"""
		ActionExplainer is a class for building a decision tree model and generating a visual
		representation of the tree for explaining the model's decision-making process.

		Args:
			variable_description (bool): A boolean flag indicating whether to use
				variable descriptions instead of variable names as feature names in the visualized
				decision tree. Defaults to True.

		Attributes:
			X (pd.DataFrame): The input feature data used for training the decision tree model.
			y (pd.Series): The output target data used for training the decision tree model.
			clf (DecisionTreeClassifier): The decision tree classifier object used for training the model.
			variables (list): A list of the input feature names used in the model.
			var_lookup_table (dict): A dictionary mapping input feature names to their descriptions
				(if `variable_description` is True) or to themselves (if `variable_description` is False).
			model (dtreeviz.DTreeViz): The decision tree visualization model.

		Returns:
			None.

		Usage:
			>> from ActionExplainer import ActionExplainer
			>> ae = ActionExplainer()
			>> ae.fit(X, y)
			>> ae.learn_data()
			>> ae.global_explanation()
			>> ae.show_prediction_path(123)
			>> ae.feature_importance(123)
			>> ae.action(123)

			Where '123' is the index of the input data point to analyse.
		"""
		self.variable_description = variable_description
	
	def fit(self, X: pd.DataFrame, y: pd.Series, max_depth: int = 4):
		"""
		Fits a decision tree model to the input data.

		Args:
			X (pd.DataFrame): The input feature data.
			y (pd.Series): The output target data.
			max_depth (int): The maximum depth of the decision tree. Defaults to 4.

		Returns:
			None.
		"""
		self.X = X
		self.y = y
		self.clf = DecisionTreeClassifier(max_depth=max_depth)
		self.clf.fit(X, y)
		if self.variable_description:
			descriptions = self._get_variable_description()
			self.variables = list(X.columns)
			
			if len(self.variables) != len(descriptions):
				raise ValueError("The length of the variables list and decriptions list are not equal.")
			else:
				self.var_lookup_table = dict(zip(self.variables, descriptions))
		else:
			self.var_lookup_table = None
	
	def learn_data(self, y_label: str = "Fault", class_names: list = ['Normal', 'Fault']):
		"""
		Generates a decision tree visualization model from the fitted decision tree classifier object.

		Args:
			y_label (str): The label for the target data in the visualization. Defaults to "Fault".
			class_names (list): A list of class names used in the visualization. Defaults to ['Normal', 'Fault'].

		Returns:
			None.
		"""
		self.model = dtreeviz.model(self.clf,
									X_train=self.X.values, y_train=self.y,
									feature_names=list(self.X.columns),
									target_name=y_label,
									class_names=class_names)
	
	def global_explanation(self, detailed: bool = True):
		"""
		Generates a visual representation of the decision tree model.

		Args:
			detailed (bool): A boolean flag indicating whether to show detailed information
				in the visualization. Defaults to True.

		Returns:
			A visualization of the decision tree model.
		"""
		if detailed:
			return self.model.view()
		else:
			return self.model.view(fancy=False)
	
	def show_prediction_path(self, index: int):
		"""
		Generates a visual representation of the decision path for a specific input data point.

		Args:
			index (int): The index of the input data point to visualize.

		Returns:
			A visualization of the decision path for the input data point.
		"""
		try:
			value = self.X.iloc[index]
		except:
			raise ValueError("Index out of range. Please check the input.")
		return self.model.view(x=value, show_just_path=True)
	
	def feature_importance(self, index: int, width: int = 8, height: int = 10):
		"""
		Generates a visualization of the feature importances for a specific input data point.

		Args:
			index (int): The index of the input data point to visualize.
			width (int): The width of the generated visualization. Defaults to 8.
			height (int): The height of the generated visualization. Defaults to 10.

		Returns:
			A visualization of the feature importances for the input data point.
		"""
		try:
			value = self.X.iloc[index]
		except:
			raise ValueError("Index out of range. Please check the input.")
		
		return self.model.instance_feature_importance(value, figsize=(width, height))
	
	def _get_variables(self):
		"""
		Returns the list of input feature names used in the model.

		Args:
			None.

		Returns:
			A list of input feature names used in the model.
		"""
		return list(self.X.columns)
	
	def _variable_lookup(self, variable: str):
		"""
		Looks up a variable in the `var_lookup_table`. If the variable is found, its value is returned.
		If the variable is not found, the variable name itself is returned.

		Args:
			variable (str): The name of the variable to look up.

		Returns:
			If the variable is found in the `var_lookup_table`, its corresponding value is returned.
			If the variable is not found, the variable name itself is returned.

		"""
		try:
			return self.var_lookup_table[variable]
		except KeyError:
			return variable
	
	def _get_variable_description(self):
		description = ['A Feed (stream 1)', 'D Feed (stream 2)', 'E Feed (stream 3)', 'A and C Feed (stream 4)',
					   'Recycle Flow (stream 8)', 'Reactor Feed Rate (stream 6)', 'Reactor Pressure', 'Reactor Level',
					   'Reactor Temperature', 'Purge Rate (stream 9)', 'Product Sep Temp', 'Product Sep Level',
					   'Prod Sep Pressure', 'Prod Sep Underflow (stream 10)', 'Stripper Level', 'Stripper Pressure',
					   'Stripper Underflow (stream 11)', 'Stripper Temperature', 'Stripper Steam Flow',
					   'Compressor Work',
					   'Reactor Cooling Water Outlet Temp', 'Separator Cooling Water Outlet Temp',
					   'Component A (stream 6)',
					   'Component B (stream 6)', 'Component C (stream 6)', 'Component D (stream 6)',
					   'Component E (stream 6)',
					   'Component F (stream 6)', 'Component A (stream 9)', 'Component B (stream 9)',
					   'Component C (stream 9)',
					   'Component D (stream 9)', 'Component E (stream 9)', 'Component F (stream 9)',
					   'Component G (stream 9)',
					   'Component H (stream 9)', 'Component D (stream 11)', 'Component E (stream 11)',
					   'Component F (stream 11)',
					   'Component G (stream 11)', 'Component H (stream 11)', 'D Feed Flow (stream 2)',
					   'E Feed Flow (stream 3)',
					   'A Feed Flow (stream 1)', 'A and C Feed Flow (stream 4)', 'Compressor Recycle Valve',
					   'Purge Valve (stream 9)',
					   'Separator Pot Liquid Flow (stream 10)', 'Stripper Liquid Product Flow (stream 11)',
					   'Stripper Steam Valve',
					   'Reactor Cooling Water Flow', 'Condenser Cooling Water Flow']
		return description
	
	def univarate_decision_boundaries(self, var: str, width: int = 8, height: int = 3):
		"""
		This method plots the decision boundaries of a univariate random forest classifier.

		Args:

		var (str): The name of the variable/column in the data to be used as the predictor variable for the classifier.
		width (int): The width of the resulting plot in inches. Default is 8.
		height (int): The height of the resulting plot in inches. Default is 3.
		Returns:

		None.
		Raises:

		ValueError: If the provided variable name is not a column in the fitted data.

		Example usage:
			  >> univariate_decision_boundaries('XMV(10)', width=10, height=5)
		"""
		try:
			rf = RandomForestClassifier(n_estimators=10, min_samples_leaf=5)
			rf.fit(self.X[var].values.reshape(-1, 1), self.y)
			decision_boundaries(rf, self.X[var], self.y, class_names=["Normal", "Fault"], feature_names=['XMV(10)'],
								target_name='Fault vs Normal', figsize=(width, height))
			plt.tight_layout()
		except KeyError:
			raise ValueError(
				"One of the two variables provided is not a column in the fitted data, please recheck the inputs.")
	
	def bivarate_decision_boundaries(self, var1: str, var2: str, width: int = 6, height: int = 6):
		"""
		Create a decision boundary plot for a given pair of features using a DecisionTreeClassifier.

		Args:
			var1 (str): The name of the first feature to use in the plot.
			var2 (str): The name of the second feature to use in the plot.

		Returns:
			None

		Raises:
			ValueError: If one of the input variables is not a column in the fitted data.

		Example:
			decision_boundaries_plot("feature1", "feature2")
		"""
		
		try:
			t = DecisionTreeClassifier(max_depth=4)
			t.fit(self.X[[var1, var2]], self.y)
			fig, ax = plt.subplots(1, 1, figsize=(width, height))
			decision_boundaries(t, self.X[[var1, var2]].values, self.y,
								feature_names=[var1, var2], target_name="Fault vs Normal",
								ax=ax, class_names=["Normal", "Fault"])
			plt.title(f"Decision Boundary")
			plt.tight_layout()
		except KeyError:
			raise ValueError(
				"One of the two variables provided is not a column in the fitted data, please recheck the inputs.")
	
	def action(self, index: int):
		"""
		This function takes an index as input and returns a dictionary of recommended actions to take
		based on whether the data at the given index is classified as an anomaly or not by a given model.

		Parameters:
		- index (int): The index of the data to be examined.

		Returns:
		- action_dict (dict): A dictionary containing recommended actions to take in order to address the anomaly,
		or a message indicating that the data is not an anomaly.

		Raises:
		- ValueError: If the index is out of range, the binary classification is not correct, or the string format is not correct.
		"""
		action_dict = {}
		try:
			value = self.X.iloc[index]
		except:
			raise ValueError("Index out of range. Please check the input.")
		
		if self.y.iloc[index] == 1:
			anomaly = True
		elif self.y.iloc[index] == 0:
			anomaly = False
		else:
			raise ValueError("Please use 0/1 Binary Classification for this.")
		
		rules = self.model.explain_prediction_path(value)
		if rules == "":
			raise ValueError("No rules found. No recommendation could be made. Please try another index")
		tokens = rules.split()
		vars = []
		try:
			for i in range(0, len(tokens), 3):
				vars.append((tokens[i], tokens[i + 1], tokens[i + 2]))
		except IndexError:
			raise ValueError("String is not in the correct format. Please check the input.\n"
							 "Correct format: '-2.74 <= XMEAS(1) -2.1 <= XMEAS(20) XMEAS(40) < 2.44 4.3 <= XMV(10)'")
		
		counter = 0
		if anomaly:
			try:
				print(
					"This index was classified as an anomaly. To try resolve this, please take the following actions:\n")
				for var in vars:
					if var[0] in self.variables:
						# Variable is smaller or equal to value
						if var[1] == '<':
							counter += 1
							action_dict["Action {}".format(
								counter)] = "{}. Increase {} component ({}) to be greater than or equal to {}. ".format(
								counter, self._variable_lookup(var[0]), var[0], var[2])
							print("{}. Increase {} component ({}) to be greater than or equal to {}. ".format(
								counter, self._variable_lookup(var[0]), var[0], var[2]))
						elif var[1] == '<=':
							counter += 1
							action_dict["Action {}".format(
								counter)] = "{}. Increase {} component ({}) to be greater than {}. ".format(
								counter, self._variable_lookup(var[0]), var[0], var[2])
							print("{}. Increase {} component ({}) to be greater than {}. ".format(
								counter, self._variable_lookup(var[0]), var[0], var[2]))
					# Variable is greater or equal to value
					elif var[2] in self.variables:
						if var[1] == '<':
							counter += 1
							action_dict["Action {}".format(
								counter)] = "{}. Decrease {} component ({}) to be less than or equal to {}.".format(
								counter, self._variable_lookup(var[2]), var[2], var[0])
							print("{}. Decrease {} component ({}) to be less than or equal to {}.".format(
								counter, self._variable_lookup(var[2]), var[2], var[0]))
						elif var[1] == '<=':
							counter += 1
							action_dict["Action {}".format(
								counter)] = "{}. Decrease {} component ({}) to be less than {}.".format(
								counter, self._variable_lookup(var[2]), var[2], var[0])
							print("{}. Decrease {} component ({}) to be less than {}.".format(
								counter, self._variable_lookup(var[2]), var[2], var[0]))
			except KeyError:
				raise ValueError("String is not in the correct format. Please check the input.\n"
								 "Correct format: '-2.74 <= XMEAS(1) -2.1 <= XMEAS(20) XMEAS(40) < 2.44 4.3 <= XMV(10)'")
		
		else:
			try:
				print("This index was classified as normal operational behaviour, the reason's for this are:\n")
				for var in vars:
					if var[0] in self.variables:
						# Variable is smaller or equal to value
						if var[1] == '<':
							counter += 1
							
							print("{}. {} component ({}) is less than {}. ".format(
								counter, self._variable_lookup(var[0]), var[0], var[2]))
						elif var[1] == '<=':
							counter += 1
							print("{}. {} component ({}) is less than or equal to {}. ".format(
								counter, self._variable_lookup(var[0]), var[0], var[2]))
					# Variable is greater or equal to value
					elif var[2] in self.variables:
						if var[1] == '<':
							counter += 1
							
							print("{}. {} component ({}) is greater than {}. ".format(
								counter, self._variable_lookup(var[2]), var[2], var[0]))
						elif var[1] == '<=':
							counter += 1
							print("{}. {} component ({}) is greater than or equal to {}. ".format(
								counter, self._variable_lookup(var[2]), var[2], var[0]))
				action_dict["Action 1"] = "Maintain systems current state"
			except KeyError:
				raise ValueError("String is not in the correct format. Please check the input.\n"
								 "Correct format: '-2.74 <= XMEAS(1) -2.1 <= XMEAS(20) XMEAS(40) < 2.44 4.3 <= XMV(10)'")
		
		return action_dict
