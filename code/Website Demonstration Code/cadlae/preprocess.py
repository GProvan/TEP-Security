import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
class DataProcessor:
    def __init__(self, train_link, test_link, y_var, drop_cols, scaler=None, imputer=None):
        '''
        :param train_link: link to the training data
        :param test_link: link to the test data
        :param y_var: name of the target variable
        :param drop_cols: list of columns to drop
        :param scaler: scaler to use for scaling the data
        :param imputer: imputer to use for imputing missing values
        '''
        df_normal = pd.read_csv(train_link)
        df_test = pd.read_csv(test_link)
        self.X_train, self.y_train = self.drop_y(df_normal, y_var)
        self.X_test, self.y_test = self.drop_y(df_test, y_var)
        self.X_train = self.drop_cols(self.X_train, drop_cols)
        self.X_test = self.drop_cols(self.X_test, drop_cols)
        self.col_names = self.get_colnames(self.X_train)
        self.X_train = self.check_and_impute_missing(self.X_train, imputer)
        self.X_test = self.check_and_impute_missing(self.X_test, imputer)
        self.X_train, self.X_test, self.scaler_function = self.scale_data(self.X_train, self.X_test, self.col_names,
                                                                          scaler)

    def drop_y(self, df, y_var):
        '''
        :param df: dataframe to drop the target variable from
        :param y_var: name of the target variable
        :return: dataframe with the target variable dropped
        '''
        y = df[y_var]
        x_data = df.drop(columns=[y_var], axis=1)
        return x_data, y

    def drop_cols(self, df, cols):
        '''
        :param df: dataframe to drop the columns from
        :param cols: list of columns to drop
        :return: dataframe with the columns dropped
        '''
        data = df.drop(columns=cols, axis=1)
        return data

    def get_colnames(self, df):
        '''
        :param df: dataframe to get the column names from
        :return: list of column names
        '''
        return list(df.columns)

    def check_and_impute_missing(self, df, imputer):
        '''
        :param df: dataframe to check for missing values
        :param imputer: imputer to use for imputing missing values
        :return: dataframe with missing values imputed
        '''
        # Check for missing values
        if imputer is None:
            imputer = SimpleImputer(strategy='mean')
        missing = df.isnull().sum()
        if missing.sum() == 0:
            # No missing values, return the original dataframe
            return df
        else:
            # Impute missing values with the mean of the column
            imputed_df = imputer.fit_transform(df)
            imputed_df = pd.DataFrame(imputed_df, columns=self.col_names)
            return imputed_df

    def scale_data(self, train_df, test_df, colnames, scaler):
        '''
        :param train_df: training dataframe to scale
        :param test_df: test dataframe to scale
        :param colnames: list of column names
        :param scaler: scaler to use for scaling the data
        :return: scaled training and test dataframes
        '''
         # if no scaler is passed, use the standard scaler
        if scaler is None:
            scaler = preprocessing.StandardScaler()
        x_scaled_train = scaler.fit_transform(train_df)
        # fit and transform the training data
        df_train = pd.DataFrame(x_scaled_train, columns=self.col_names)
        # transform the test data
        x_scaled_test = scaler.transform(test_df)
        df_test = pd.DataFrame(x_scaled_test, columns=self.col_names)
        return df_train, df_test, scaler