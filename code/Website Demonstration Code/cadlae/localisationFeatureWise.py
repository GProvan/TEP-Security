import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
class FeatureWiseLocalisation:
    def __init__(self, y_test, y_pred, col_names, training_error, test_error):
        self.y_test = y_test
        self.y_pred = y_pred
        self.col_names = col_names
        self.training_error = training_error
        self.test_error = test_error

    def find_optimal_threshold(self):
        '''
        :param y_test: test target variable
        :param y_pred: predicted target variable
        :return: optimal threshold
        '''
        # get the false positive rate, true positive rate and thresholds
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred)
        # get the optimal threshold based on the maximum tpr - fpr
        optimal_idx = np.argmax(tpr - fpr)
        # get the optimal threshold
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold

    def convert_scores_to_label(self):
        '''
        The function converts the scores to labels based on the threshold, if the score is greater than the threshold,
        the label is 1, else 0
        :param array: array of scores
        :param threshold: threshold to use for converting the scores to labels
        :return: array of labels
        '''
        threshold = self.find_optimal_threshold()
        binary = []
        for i in self.y_pred:
            if i < threshold:
                binary.append(0)
            else:
                binary.append(1)
        return binary

    def get_indices(self):
        '''
        :param y_pred: predicted target variable
        :return: indices of the positive and negative predictions
        '''
        y_predictions = self.convert_scores_to_label()
        normal_idx, anomaly_idx = [], []
        for i in range(len(y_predictions)):
            if y_predictions[i] == 0:
                normal_idx.append(i)
            else:
                anomaly_idx.append(i)
        return normal_idx, anomaly_idx

    def normal_vs_anomaly_values(self, array, normal_idx):
        '''
        Get values of normal data points
        :param array: array to get values from
        :param index_list: list of indices
        :return: array of values
        '''
        normal_values = []
        anomaly_values = []
        for i in range(len(array)):
            if i in normal_idx:
                normal_values.append(array[i])
            else:
                anomaly_values.append(array[i])
        return normal_values, anomaly_values

    def max_reconstruction_error(self):
        '''
        :param column_names: list of column names
        :param details: dictionary of details (reconstruction error, mean, std)
        :return: dataframe with the max reconstruction error
        '''
        max_recon = {}
        for i in range(0, len(self.col_names)):
            col_name_iter = self.col_names[i]
            max_recon[col_name_iter] = max(self.training_error['errors_mean'][i])
        return max_recon

    def threshold_violations(self):
        '''
        :param column_names: list of column names
        :param details: dictionary of details (reconstruction error, mean, std)
        :param normal_val_index: indices of normal values
        :return: dataframe with the threshold violations
        '''
        normal_index, anomaly_index = self.get_indices()
        max_recon_train = self.max_reconstruction_error()
        count_violations = {}
        for i in range(0, len(self.col_names)):
            count = 0
            col_name_iter =self.col_names[i]
            normal_vals, anomaly_values = self.normal_vs_anomaly_values(self.test_error['errors_mean'][i], normal_index)
            for i in anomaly_values:
                # if the reconstruction error is greater than the threshold, increment the count
                if i > max_recon_train[col_name_iter]:
                    count += 1
            # get the percentage of threshold violations
            try:
                count_violations[col_name_iter] = (count, round((count / len(anomaly_values)) * 100, 2))
            except ZeroDivisionError:
                count_violations[col_name_iter] = (count, 0)
        return count_violations

    def run(self):
        '''
        Sort dictionary by value
        :param dictionary: dictionary to sort
        :return: sorted dictionary
        '''
        dictionary = self.threshold_violations()
        sorted_dict = sorted(dictionary.items(), key=lambda x: x[1][0], reverse=True)
        y_predictions = self.convert_scores_to_label()
        return sorted_dict, y_predictions
