def get_attacks(y_test, outlier=1, normal=0, breaks=[]):
    '''
    Get indices of anomalies
    :param y_test: predictions from semi supervised model
    :param outlier: label for anomalies
    :param normal: label for normal data points
    :param breaks: indices of breaks in data
    :return:
    '''
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
            elif tim in breaks:
                # A break point was hit, end current event and start new one
                event_end = tim - 1
                events[event] = (event_start, event_end)
                event += 1
                event_start = tim

        else:
            # event_by_time_true[tim] = 0
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events

#dict_attacks = get_attacks(y_test, outlier=1, normal=0, breaks=[])
#dict_attacks1 = get_attacks(y_predictions, outlier=1, normal=0, breaks=[])

def get_attack_idx_list(dictionary):
    '''
    Get list of indices of anomalies
    :param dictionary: dictionary of anomalies
    :return: Dictionary of anomalies, value is changed from (start, end) to list of indices
    '''
    for key, value in dictionary.items():
        if isinstance(value, tuple):
            dictionary[key] = list(range(value[0], value[1] + 1))
    return dictionary

class Anchors:
    def __init__(self, model, X_train, y_train, X_test, y_pred, indices, norm_to_anom_ratio, max_anchors):
        '''
        Anchor explanation for anomalies
        :param model: model to be used for classification
        :param X_train: training data
        :param y_train: training labels
        :param X_test: test data
        :param y_pred: predicted labels from semi supervised model
        :param indices: indices of data points to be explained
        :param norm_to_anom_ratio: ratio of normal to anomalous data points to be sampled
        :param max_anchors: maximum number of anchors to be used

        Both sampling and max_anchors are used to reduce computation time / expense
        The more anchors, the more accurate the higher the precision and coverage
        The higher the ratio, the more accurate the higher the precision and coverage
        TODO:
          - Add functionality to get link predictions to Anchor, so that we can get attack runs quickly
          - Add ability to skip sampling -> use all training data
          - Need error handling
          - Add recommendations i.e. Ratios / num of anchors (need to experiment)

        '''
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_pred = y_pred
        self.indices = indices
        self.norm_to_anom_ratio = norm_to_anom_ratio
        self.max_anchors = max_anchors

    def sample_dataset(self):
        '''
        Sample normal and anomalous data points
        :return:
        '''
        # number of normal data points to sample
        ratio = len(self.indices) * self.norm_to_anom_ratio
        # index reset
        self.X_train.reset_index(inplace=True, drop=True)
        self.y_train.reset_index(inplace=True, drop=True)
        # sampling
        X_sample = self.X_train.sample(ratio)
        y_sample = self.y_train[X_sample.index]
        # catch instances where we are checking one point
        if len(self.indices) == 1:
            start = self.indices[0]
            end = start + 1
        else:
            start = self.indices[0]
            end = self.indices[-1]
        # append normal and anomalous data points
        df_x = X_sample.append(self.X_test.iloc[start:end + 1], ignore_index=True)
        df_y = y_sample.append(self.y_pred.iloc[start:end + 1], ignore_index=True)
        return df_x, df_y

    def train(self, X, y):
        '''
        Train model on sampled data
        :param X: sampled training data + anomalies
        :param y: sampled training labels + anomalies
        :return: trained model
        '''
        return self.model.fit(X, y)

    def check_predictions(self, X, y_test):
        '''
        Check predictions of model
        :param X: sampled training data + anomalies
        :param y_test: sampled training labels + anomalies
        '''
        y_pred = self.model.predict(X)
        # check to make sure predictions are correct
        print("Clf model accuracy: [{:.4f}]".format(sklearn.metrics.accuracy_score(y_test, y_pred)))

    def build_explainer(self, X):
        '''
        Build anchor explainer
        :param X: sampled training data + anomalies
        :return: anchor explainer
        '''
        # predict_fn takes as input a batch and returns the prediction probabilities for the batch
        predict_fn = lambda x: self.model.predict_proba(x)
        explainer = AnchorTabular(predict_fn, list(self.X_test.columns))
        return explainer.fit(X.to_numpy(), disc_perc=(25, 50, 75))

    def average_reason(self, values):
        # Initialize variables to store the sum of values and the count of values
        total_sum = 0
        count = 0
        # Initialize a Counter to count the occurrences of each sign
        sign_counts = Counter()

        # Iterate through the values
        for value in values:
            # Split the value into the sign and the number
            sign, number = value[0], float(value[1:])
            # Add the number to the total sum and increment the count
            total_sum += number
            count += 1
            # Increment the count for the sign in the Counter
            sign_counts[sign] += 1

        # Calculate the average value
        average = total_sum / count
        # Get the most common sign and its count
        most_common_sign, sign_count = sign_counts.most_common(1)[0]

        return f'{most_common_sign} {average:.2f}'

    def explain(self):
        fault_dict = {}  # dictionary to store explanations
        reasoning_dict = {}
        df_x, df_y = self.sample_dataset()
        self.train(df_x, df_y)
        self.check_predictions(df_x, df_y)
        explainer = self.build_explainer(df_x)
        anomaly_idx = np.where(df_y == 1)[0]  # index of anomalies
        # tqdm is used to show progress bar
        for i in (range(len(self.indices))):
            # get explanation for each anomaly
            explanation = explainer.explain(df_x.to_numpy()[anomaly_idx[i]], threshold=0.90,
                                            max_anchor_size=self.max_anchors)

            print(str(self.indices[i]) + ' Anchor: %s' % (' AND '.join(explanation.anchor)))
            for i in explanation.anchor:
                if i.split()[0] in fault_dict:  # if feature is already in dictionary
                    fault_dict[i.split()[0]] += 1
                    reason = i.split()[1:]
                    reason = " ".join(reason)
                    reasoning_dict[i.split()[0]].append(reason)
                else:
                    fault_dict[i.split()[0]] = 1
                    reason = i.split()[1:]
                    reason = " ".join(reason)
                    reasoning_dict[i.split()[0]] = [reason]

        print("Fault is most likely related to: ")
        print('\n')
        # sort dictionary by value
        likely_faults = (sorted(fault_dict.items(), key=lambda item: item[1], reverse=True))
        most_likely = likely_faults[0][1]
        for i in likely_faults:
            # only print faults that are at least 1/3 of the most likely fault
            if i[1] > most_likely / 3:
                reason = self.average_reason(reasoning_dict[i[0]])
                print("Feature: {}".format(i[0]))
                print('Reason: {}'.format(reason))
                print("Appeared in samples examined: {}/{}".format(str(i[1]), str(len(self.indices))))
                print("\n")



