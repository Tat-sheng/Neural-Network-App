import streamlit as st

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow import keras

from numpy.random import seed
seed(100)

import matplotlib.pyplot as plt
from IPython.display import clear_output
# import packages
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, average_precision_score


# NN packages
import os
import time

import scipy as sci
import scipy.io as sio

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow import keras

from numpy.random import seed
seed(100)

from IPython.display import clear_output

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn import utils


import warnings
warnings.filterwarnings("ignore")


st.title("Model Pipeline Builder")
st.write("Using the learnings from the previous excercises, this page allows a user to upload a csv file \
         on a large dataset, and allows them to choose predictor and target variables \
         from the file to automatically tune and select the best model on the dataset.")

file = st.file_uploader("Upload your csv here")

#=================================================================================================================================

def extract_integer(value):
    try:
        value_str = str(value)
        if any(char.isdigit() for char in value_str):
            return float(''.join(filter(str.isdigit, value_str)))
        else:
            return value
    except:
        return 0

def get_unique_items(df_column):
    unique_items = pd.unique(df_column)
    return unique_items

def converter(list, list2,  x):
    if x in list:
        return 0
    elif x in list2:
        return 1
    else:
        return 0
    
#==============================================================================================================================

# Logistic Regression
def logistic_regression_model(X_train, X_test, y_train, y_test, param_grid):

    # Create a logistic regression object
    lr = LogisticRegression()

    # Create a GridSearchCV object
    grid_lr = GridSearchCV(lr, param_grid, cv=5, scoring=['roc_auc', 'f1', 'precision'], refit='roc_auc')

    # Fit the GridSearchCV object on the training data
    grid_lr.fit(X_train, y_train)


    # Compute the F1 score for the best model on the testing data
    y_pred = grid_lr.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Plot ROC curve
    y_prob = grid_lr.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    print('Best Params: ', grid_lr.best_params_)

    plt.figure(figsize=(8,6))      # format the plot size
    lw = 1.5
    plt.plot(fpr, tpr, color='darkorange', marker='.',
            lw=lw, label='Logistic Regression (AUC = %0.4f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
            label='Random Prediction (AUC = 0.5)' )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

    # Return the AUC ROC plot and performance measures
    return [auc_score, f1, precision]

lr_param_grid = {'penalty': ['l1', 'l2'],
                'C': np.logspace(-3, 3, 7),
                'solver': ['liblinear'],
                'fit_intercept': [True, False],
                'max_iter': [100, 500, 1000],
                'tol': [1e-4, 1e-3, 1e-2]}

# Random Forest
def random_forest_classifier(X_train, X_test, y_train, y_test, param_grid):
    # Create Random Forest Classifier object
    rf = RandomForestClassifier()
    grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring=['roc_auc', 'f1', 'precision'], refit='roc_auc')
    grid_rf.fit(X_train, y_train)

    # Obtain performance measures
    y_pred = grid_rf.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    # Plot ROC curve
    y_prob = grid_rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    print('Best Params: ', grid_rf.best_params_)

    plt.figure(figsize=(8,6))      # format the plot size
    lw = 1.5
    plt.plot(fpr, tpr, color='darkorange', marker='.',
            lw=lw, label='Random Forest (AUC = %0.4f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
            label='Random Prediction (AUC = 0.5)' )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

    
    # Return the AUC ROC plot and performance measures
    return [auc_score, f1, precision]

rf_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]}


def xgboost_classifier(X_train, X_test, y_train, y_test, param_grid):

    # Create a XGB Classifier object
    xgb_ = xgb.XGBClassifier()
    grid_xgb = GridSearchCV(xgb_, param_grid, cv=5, scoring=['roc_auc', 'f1', 'precision'], refit='roc_auc')
    grid_xgb.fit(X_train, y_train)
    
    # Obtain performance measures
    y_pred = grid_xgb.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    # Plot ROC curve
    y_prob = grid_xgb.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    print('Best Params: ', grid_xgb.best_params_)

    plt.figure(figsize=(8,6))      # format the plot size
    lw = 1.5
    plt.plot(fpr, tpr, color='darkorange', marker='.',
            lw=lw, label='XGBoost Classifier (AUC = %0.4f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
            label='Random Prediction (AUC = 0.5)' )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

    # Return the AUC ROC plot and performance measures
    return [auc_score, f1, precision]

xg_param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 500, 1000]}


models_ = [('lrm', LogisticRegression()), ('rfm', RandomForestClassifier()), ('xgbm', xgb.XGBClassifier())]

def ensemble_model(X_train, X_test, y_train, y_test, params):

    # Define models
    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier()
    clf3 = xgb.XGBClassifier()
    estimators = [
                ('lr', clf1),
                ('rf', clf2),
                ('xgb', clf3)]
    eclf = VotingClassifier(estimators=estimators)
    grid = GridSearchCV(estimator=eclf, param_grid=params,verbose=2, cv=5,n_jobs=-1)
    grid.fit(X_train, y_train)

    # grid_search.fit(X_train, y_train)

    # Evaluate on test data
    y_pred = grid.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Plot ROC curve
    plt.figure(figsize=(8,6))      # format the plot size
    lw = 1.5
    plt.plot(fpr, tpr, color='darkorange', marker='.',
            lw=lw, label='Ensemble Model (AUC = %0.4f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
            label='Random Prediction (AUC = 0.5)' )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

    # Return the AUC ROC plot and performance measures
    return [auc_score, f1, precision]


# Define the models to include in the ensemble
en_param_grid = {
    'lr__C': [0.1, 1.0, 10.0],
    'rf__n_estimators': [50, 100, 200],
    'xgb__learning_rate': [0.1, 0.01, 0.001],
    'xgb__max_depth': [3, 5, 7]
}


import kerastuner.tuners as kt


def remove_suffix(string):
    """
    Removes a suffix from a string.
    
    Args:
        string (str): A string.
    
    Returns:
        A string with suffix removed.
    """
    if "_" in string:
        parts = string.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            new_string = "_".join(parts[:-1])
        else:
            new_string = string
    else:
        new_string = string
    return new_string



def fit_nn_model(X_train, X_test, y_train, y_test):

        # Divide training set into training and validation set
        X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.5,random_state=100)

        tuner = kt.Hyperband(
        model_builder, 
        objective = 'loss',
        executions_per_trial = 3,
        directory = '6.18',
        project_name='LendingClubNN'
        )


        # Tune on train/val data
        tuner.search(X_train, 
                y_train, 
                epochs=50, 
                validation_data=(X_val, y_val))

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(X_train, y_train, epochs=100, validation_split=0.5, verbose=0)

        auc = 0

        for i in (history.history.keys()):
                if remove_suffix(i) == 'val_auc':
                     print(remove_suffix(i))
                     auc = history.history[i]

        # Tune model based on validation accuracy
        auc_per_epoch = auc
        best_epoch = auc_per_epoch.index(max(auc)) + 1


        # # Reinstantiating models from above
        hypermodel_btc = tuner.hypermodel.build(best_hps)
        hypermodel_btc.fit(X_train, y_train, epochs=best_epoch, validation_split=0.5, verbose =0)

        # Make predictions on the test data
        y_pred_prob = model.predict(X_test)

        # Compute the performance measures
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        f1 = f1_score(y_test, y_pred_prob > 0.5)
        precision = precision_score(y_test, y_pred_prob > 0.5)
        
        print(best_hps)

        plt.figure(figsize=(8,6))      # format the plot size
        lw = 1.5
        plt.plot(fpr, tpr, color='darkorange', marker='.',
                lw=lw, label='NN Model (AUC = %0.4f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
                label='Random Prediction (AUC = 0.5)' )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.show()

        return auc_score, f1, precision


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(model_len, )))

    # Tune the number of units in the first Dense layer, optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='sigmoid'))
    model.add(keras.layers.Dense(1))

    # Tune the learning rate for the optimizer, optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss= 'mae',
                    metrics=[tf.keras.metrics.AUC(from_logits=True),tf.keras.metrics.Precision(thresholds=0)])

    return model
# ===============================================================================================================================

if file:
    df = pd.read_csv(file)
    col = df.columns

    # Drop all columns with more than 75% NaN values
    threshold = len(df) * 0.75
    df = df.dropna(thresh = threshold, axis = 1)

    # Get new column names
    nc = df.columns

    y_choice = st.selectbox("Choose one binary target variable - Y", nc)

    unique_items = np.unique(df[y_choice]).tolist()
        
    non_numeric_cols = df.select_dtypes(exclude=[int, float]).columns.tolist()

    for i in non_numeric_cols:
        df[i] = df[i].apply(lambda x: extract_integer(x))

    modelnames = ['Logistic Regression', 'Random Forest', 'XGBoost', \
                'Ensemble Classifier (LR, RF, XG)', 'Neural Network']
    
    model_choice = st.selectbox("Select the model that you want to use", modelnames)
    
    # with st.form(key = "select cols"):
    class1= st.multiselect("Select all that you want to group to Bin 0", unique_items)

    # make sure you don't bin the same values twice
    not_selected = [x for x in unique_items if x not in class1]
    class2 = st.multiselect("Select all that you want to group to Bin 1", not_selected
                            )


    confirm1 = st.button("Confirm")

    if confirm1:
        with st.spinner("Please wait while the model calculates the results..."):
            df[y_choice] = df[y_choice].apply(lambda x: converter(class1, class2, x))
            df = df.replace(np.nan, 0)
            y_ = df[y_choice]

            df = df.drop(y_choice, axis = 1)
            X_ = pd.get_dummies(df, drop_first= True)

            X_train,X_test,y_train,y_test=train_test_split(X_,y_,test_size=0.5,random_state=100)

            model_len = len(X_train.columns.tolist())

            if model_choice == 'Logistic Regression':
                auc_score, f1, precision = (logistic_regression_model(X_train, X_test, y_train, y_test, lr_param_grid))
            elif model_choice == 'Random Forest':
                auc_score, f1, precision = (random_forest_classifier(X_train, X_test, y_train, y_test, rf_param_grid)) 
            elif model_choice == 'XGBoost':
                auc_score, f1, precision = (xgboost_classifier(X_train, X_test, y_train, y_test, xg_param_grid)) 
            elif model_choice == 'Ensemble Classifier (LR, RF, XG)':
                auc_score, f1, precision = (ensemble_model(X_train, X_test, y_train, y_test, en_param_grid)) 
            elif model_choice == 'Neural Network':
                auc_score, f1, precision = (fit_nn_model(X_train, X_test, y_train, y_test)) 

            st.write("Model results:")
            st.write("AUC Score: ", auc_score)
            st.write("F1 Score: ", f1)
            st.write("Precision Score: ", precision)
