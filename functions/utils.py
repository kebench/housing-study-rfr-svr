import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def removeNonUniqueColumns(dataframe):
    # Remove the non unique columns
    # https://stackoverflow.com/questions/50582168/pandas-get-all-columns-that-have-constant-value
    non_unique_columns = dataframe.columns[dataframe.nunique() <= 1]
    dataframe = dataframe.drop(non_unique_columns, axis=1)
    return dataframe

def extractIrelandDataAndTranspose(dataframe):
    dataframe = dataframe[dataframe["Country Name"] == "Ireland"]
    new_dataframe = dataframe.drop(dataframe.columns[np.r_[0:19,61:67]], axis=1)
    new_dataframe = new_dataframe.T.reset_index()
    new_dataframe.iloc[:, 0] = new_dataframe.iloc[:, 0].astype(int)
    return new_dataframe

def displayRegressionMetrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("Root Mean Squared Error:", mse**.5)
    print('Mean Squared Error:', mse)
    print('Mean Absolute Error:', mae)
    print('R2 Score:', r2)
    
def plotFeatureImportance(feat_imp, dataframe):
    # Access the most important features
    feature_imp = pd.Series(feat_imp,index=dataframe.columns).sort_values(ascending=False)    
    plt.figure(figsize=(20,12))
    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    #plt.legend()
    plt.show()

def plotSVRFeatureImportance(coefficient, X):
    feature_importance = np.abs(coefficient)
    feature_names = X.columns.tolist()
    normalized_feature_importance = feature_importance / np.sum(feature_importance)
    # Create a dictionary to store feature importance values
    feature_importance_dict = dict(zip(feature_names, normalized_feature_importance))

    # Sort the feature importance dictionary by value in descending order
    sorted_feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

    # Extract the sorted feature names and their corresponding importance values
    sorted_feature_names = list(sorted_feature_importance_dict.keys())
    sorted_feature_importance = list(sorted_feature_importance_dict.values())
    plt.figure(figsize=(20,12))
    # Creating a bar plot
    sns.barplot(x=sorted_feature_importance, y=sorted_feature_names)
    # plt.barh(range(len(sorted_feature_names)), sorted_feature_importance)
    # plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
    plt.xlabel('Normalized Feature Importance')
    plt.title('SVR - Feature Importance')
    plt.show()