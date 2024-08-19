import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def plot_anomalies(df, column, anomalies,scaler = None):
    '''
    Plot anomalies
    :param df: dataframe to plot
    :param column: column to plot
    :param anomalies: dictionary of anomalies -> pass through dictionary - list pipeline to get dictionary with indx of anomalies
    :param reverse_scaler: object used to scale the data -> reverses to original scale in plot of passed
    '''
    if scaler is not None:
      df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
    title = "Plot of {}".format(column)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df[column])
    ax.set_title(title)
    for key, value in anomalies.items():
        ax.plot(value, df[column][value], 'ro', markersize=4)
    plt.show()


#fig, ax = plt.subplots(13, 4, figsize=(40, 90))
def plot_all_features(df):
    for i in range(df.shape[1]):
        x = df.iloc[:, i]

        mean = x.mean()
        std = x.std(ddof=1)

        LCL = mean - 3 * std
        UCL = mean + 3 * std

        x.plot(ax=ax.ravel()[i])

        ax.ravel()[i].legend()

        ax.ravel()[i].axhline(mean, c='k')
        ax.ravel()[i].axhline(LCL, ls='--', c='r')
        ax.ravel()[i].axhline(UCL, ls='--', c='r')


#plot_all_features(X_test)


#fig, ax = plt.subplots(X_train.shape[1], 2, figsize=(30, 400))
def plot_all_features_train_vs_test(xtrain, xtest):
    # turn into function , add the same y axis for each
    # df2 = df2.drop('Unnamed: 0', axis=1)
    for i in range(xtest.shape[1]):
        x = xtrain.iloc[:, i]
        k = xtest.iloc[:, i]

        ymin = min(min(x), min(k))

        ymax = max(max(k), max(x))

        mean = x.mean()
        std = x.std(ddof=1)

        LCL = mean - 3 * std
        UCL = mean + 3 * std

        x.plot(ax=ax[i, 0])

        ax[i, 0].legend()

        ax[i, 0].axhline(mean, c='k')
        ax[i, 0].axhline(LCL, ls='--', c='r')
        ax[i, 0].axhline(UCL, ls='--', c='r')
        ax[i, 0].set_ylim([ymin, ymax])

        mean = k.mean()
        std = k.std(ddof=1)

        LCL = mean - 3 * std
        UCL = mean + 3 * std

        k.plot(ax=ax[i, 1])

        # ax[1].legend()

        ax[i, 1].axhline(mean, c='k')
        ax[i, 1].axhline(LCL, ls='--', c='r')
        ax[i, 1].axhline(UCL, ls='--', c='r')
        ax[i, 1].set_ylim([ymin, ymax])


#plot_all_features_train_vs_test(X_train, X_test)




#fig, ax = plt.subplots(1,2,figsize=(30,10))
def side_by_side_time_plot(xtrain, xtest, feature_name):
    i = list(xtrain.columns).index("XMV(10)")
    x = xtrain.iloc[:,i]

    mean  = x.mean()
    std = x.std(ddof=1)

    LCL = mean-3*std
    UCL = mean+3*std

    x.plot(ax = ax[0])

    ax[0].legend()

    ax[0].axhline(mean,c='k')
    ax[0].axhline(LCL,ls='--',c='r')
    ax[0].axhline(UCL,ls='--',c='r')

    k = xtest.iloc[:,i]


    mean  = k.mean()
    std = k.std(ddof=1)

    LCL = mean-3*std
    UCL = mean+3*std

    k.plot(ax = ax[1])

    #ax[1].legend()

    ax[1].axhline(mean,c='k')
    ax[1].axhline(LCL,ls='--',c='r')
    ax[1].axhline(UCL,ls='--',c='r')
#side_by_side_time_plot(X_train, X_test, "XMV(10)")

#fig, ax = plt.subplots(1,2,figsize=(30,10))
def side_by_side_time_plot_equal_scale(normal, anomaly, col_name):
    def compare_plot_helper(data, title, index, ax_val, ymin, ymax):
        col_name = data.columns[index]
        df = data.iloc[:, index]
        mean = df.mean()
        std = df.std(ddof=1)
        LCL = mean - 3 * std
        UCL = mean + 3 * std
        ax[ax_val].legend()
        ax[ax_val].axhline(mean, c='k')
        ax[ax_val].axhline(LCL, ls='--', c='r')
        ax[ax_val].axhline(UCL, ls='--', c='r')
        ax[ax_val].set_title(title + col_name, fontsize=25)
        ax[ax_val].set_ylim([ymin, ymax])
        for item in (ax[ax_val].get_xticklabels() + ax[ax_val].get_yticklabels()):
            item.set_fontsize(15)

    def compare(normal, anomaly, col_name):
        index = list(X_train.columns).index(col_name)
        ymin = min(min(anomaly.iloc[:, index]), min(normal.iloc[:, index]))
        ymin = ymin - abs(ymin * 0.1)

        ymax = max(max(anomaly.iloc[:, index]), max(normal.iloc[:, index]))
        ymax = ymax + abs(ymax * 0.1)
        compare_plot_helper(normal, "Normal Data for ", index, 0, ymin, ymax)
        compare_plot_helper(anomaly, "Fault Data for ", index, 1, ymin, ymax)

    compare(normal, anomaly, col_name)


#side_by_side_time_plot_equal_scale(X_train, X_test, "XMV(10)")

def plot_reconstructions_errors(det, X, column, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    col_idx = X.columns.get_loc(column)
    ax.plot(det['errors_mean'][col_idx], label='reconstructed error mean')
    ax.set_title('Reconstructions Error of column {}'.format(column))
    ax.legend()
    return ax

def plot_reconstructions_mean(det, X, column,yaxis=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    col_idx = X.columns.get_loc(column)
    if yaxis is not None:
        ax.axis(ymin=yaxis[0],ymax=yaxis[1])
    ax.plot(det['reconstructions_mean'][col_idx], label='reconstructed error mean')
    ax.set_title('Reconstructions Error of column {}'.format(column))
    ax.legend()
    return ax

def plot_reconstructions(details, X, column, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(X[column], label='original series')
    col_idx = X.columns.get_loc(column)
    ax.plot(details['errors_mean'][col_idx], label='reconstructed error mean')
    ax.set_title('Reconstructions of column {}'.format(column))
    ax.legend()
    return ax

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def ROC(y_test,y_pred):
    fpr,tpr,tr=roc_curve(y_test,y_pred)
    auc=roc_auc_score(y_test,y_pred)
    idx=np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot(fpr,1-fpr,'r:')
    plt.plot(fpr[idx],tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]

def make_confusion_matrix(y_true, y_prediction, normalise = False, c_map = "viridis"):
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_prediction)
    format = "d"
    if normalise== True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        format = '.4f'
    cm_matrix = pd.DataFrame(data=cm, columns=['Normal', 'Attack'],
                                index=['Normal', 'Attack'])

    sns.heatmap(cm_matrix, annot=True, fmt=format, cmap=c_map,linewidths=1, linecolor='black',clip_on=False)
    plt.show()

#make_confusion_matrix(y_test,y_pred_binary)

'''
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
df = results_good_cm
methods = {"Isolation Forest": IsolationForestSKI,
           "Autoencoder":AutoEncoderSKI,
           "LSTM": LSTMODetectorSKI,
           "OCSVM": OCSVMSKI,
           "Variational AutoEncoder": VariationalAutoEncoderSKI,
           "AutoRegressive" : AutoRegODetectorSKI,
           "KNN": KNNSKI,
           "Proposed Model": LSTMED}

predictions = {"Isolation Forest": df['Isolation Forest'],
           "Autoencoder":df['Autoencoder'],
           "LSTM": df["LSTM"],
           "OCSVM": df[ "OCSVM"],
           "Variational AutoEncoder": df["Variational AutoEncoder"],
           "AutoRegressive" : df["AutoRegressive"],
           "KNN": df["KNN"],
           "Proposed Model": df["Proposed Model"] }

axis_counter = {0:[0,0],
                1:[0,1],
                2:[1,0],
                3:[1,1],
                4:[2,0],
                5:[2,1],
                6:[3,0],
                7:[3,1]}
'''

def confusion_matrix_all_models(y_true, classifiers, predictions, axis_setter,normalise = False, nrows=2, ncols=4, width=10, height=20,
                                c_map="viridis"):
    f, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharex='col', sharey='row')
    for i, (key, classifier) in enumerate(classifiers.items()):
        y_pred = predictions[key]
        cf_matrix = confusion_matrix(y_true, y_pred)

        if normalise== True:
            cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

        disp = ConfusionMatrixDisplay(cf_matrix,
                                      display_labels=['Normal', 'Attack'])
        disp.plot(ax=axes[axis_setter[i][0]][axis_setter[i][1]], cmap=c_map)
        disp.ax_.grid(False)
        disp.ax_.set_title(key)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i != 0:
            disp.ax_.set_ylabel('')

    f.text(0.44, 0.05, 'Predicted label', ha='center')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    f.colorbar(disp.im_, ax=axes)
    plt.show()


#confusion_matrix_all_models(df["y_true"], methods, predictions, axis_counter,True, 4, 2, 15, 20)


def make_confusion_matrix(y_true, y_prediction, normalise = False, c_map = "viridis"):
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    ax = plt.axes()
    ax.set_title('lalala')
    cm = confusion_matrix(y_true, y_prediction)
    format = "d"
    if normalise== True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        format = '.4f'
    cm_matrix = pd.DataFrame(data=cm, columns=['Normal', 'Attack'],
                                index=['Normal', 'Attack'])

    sns.heatmap(cm_matrix, annot=True, fmt=format, cmap=c_map,linewidths=1, linecolor='black',clip_on=False)
    plt.show()

#make_confusion_matrix(y_test,y_pred_binary)

'''
#y_true = y_test[10:len(aa)]
#y_scores = aa[10:len(aa)]
y_true = y_test
y_scores = test_preds

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
print(roc_auc_score(y_true, y_scores))
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Threshold value is:", optimal_threshold)
plot_roc_curve(fpr, tpr)
'''