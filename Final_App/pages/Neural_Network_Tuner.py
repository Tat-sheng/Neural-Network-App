import streamlit as st

import numpy as np
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow import keras

from numpy.random import seed
seed(100)

import matplotlib.pyplot as plt
from IPython.display import clear_output


import warnings
warnings.filterwarnings("ignore")
# ---------------------

st.title("Options Data Model")
st.write("This page allows a user to upload a csv file on options data, and allows them to choose predictor and target variables from the file to manually tune the model using table metrics.")

file = st.file_uploader("Upload options csv here")

if file:
    df = pd.read_csv(file)
    st.write("Your uploaded data:")
    st.dataframe(df)
    col = df.columns


    y_choice = st.multiselect("Choose target variables - Y", col)
    x_choice = st.multiselect("Choose predictor variables - X", col)

    confirm = st.button("Confirm")

    if confirm and len(y_choice) == 2:
        y = df[y_choice]
        x = df[x_choice]

        with st.spinner("Please wait while the model calculates the best parameters"):
            X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=0.5,random_state=100)

            # Divide training set into training and validation set
            X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.5,random_state=100)

            # Scale features based on Z-Score
            scaler = StandardScaler()
            scaler.fit(X_train)


            X_scaled_train = scaler.transform(X_train)
            X_scaled_vals = scaler.transform(X_val)
            X_scaled_test = scaler.transform(X_test)
            y_train = np.asarray(y_train)
            y_val = np.asarray(y_val)
            y_test = np.asarray(y_test)

            # Base model
            model = keras.models.Sequential([Dense(20,activation = "sigmoid",input_shape = (6,)),
                                            Dense(20,activation = "sigmoid"),Dense(20,activation = "sigmoid"),
                                            Dense(1)])

            # Changing the number of hidden layers
            two_layers = keras.models.Sequential([Dense(20,activation = "sigmoid",input_shape = (6,)),
                                            Dense(20,activation = "sigmoid"),Dense(20,activation = "sigmoid"),
                                            Dense(20,activation = "sigmoid"),Dense(20,activation = "sigmoid"),
                                            Dense(1)])

            three_layers = keras.models.Sequential([Dense(20,activation = "sigmoid",input_shape = (6,)),
                                            Dense(20,activation = "sigmoid"),Dense(20,activation = "sigmoid"),
                                            Dense(20,activation = "sigmoid"),Dense(20,activation = "sigmoid"),
                                            Dense(20,activation = "sigmoid"),Dense(20,activation = "sigmoid"),
                                            Dense(1)])

            # Changing the number of neurons
            fewer_neurons = keras.models.Sequential([Dense(10,activation = "sigmoid",input_shape = (6,)),
                                            Dense(10,activation = "sigmoid"),Dense(10,activation = "sigmoid"),
                                            Dense(1)])

            more_neurons = keras.models.Sequential([Dense(10,activation = "sigmoid",input_shape = (6,)),
                                            Dense(10,activation = "sigmoid"),Dense(10,activation = "sigmoid"),
                                            Dense(1)])

            # List for all the models
            models = [model, two_layers, three_layers, fewer_neurons, more_neurons]

            # Complie function allows you to choose your measure of loss and optimzer
            for mdl in models:
                mdl.compile(loss = "mae",optimizer = "Adam")

            # Create checkpoints abd early stop files for each model
            checkpoints, early_stops = [], []

            for n, mdl in enumerate(models, 1):
                checkpoints.append(keras.callbacks.ModelCheckpoint("Model " + str(n) + ".h5",save_best_only = True))
                early_stops.append(keras.callbacks.EarlyStopping(patience = 50,restore_best_weights = True))
                
            # Training with callback
            history_ = []

            for ind, mdl in enumerate(models, 0):
                history_.append(mdl.fit(X_scaled_train,y_train[:,0],epochs= 100,verbose = 0, validation_data=(X_scaled_vals,y_val[:,0]), callbacks=[checkpoints[ind],early_stops[ind]]))

            # Load the best models saved and calcuate MAE for testing set for each case
            model_names = ["Model " + str(i) for i in range(1,6)]
            mae_test_vals, mean_error, std_error, mean_error_vs_BS_price, std_error_vs_BS_price, BS_mean_error, BS_std_error = [],[],[],[],[],[],[]

            for ind, mdl in enumerate(models, 0):
                temp_model = keras.models.load_model(str(model_names[ind])+".h5")
                mae_test_vals.append(mdl.evaluate(X_scaled_test,y_test[:,0],verbose=0))

                # Obtain each models predictions
                model_prediction = mdl.predict(X_scaled_test)

                # Black Scholes Statistics
                mean_error.append(np.average(model_prediction.T - y_test[:,0])) # Mean error on test set
                std_error.append(np.std(model_prediction.T - y_test[:,0])) # Standard deviation of error on test set

                # Neural Network Statistics
                mean_error_vs_BS_price.append(np.average(model_prediction.T - y_test[:,1])) # Mean error on test set vs. option price with noise
                std_error_vs_BS_price.append(np.std(model_prediction.T - y_test[:,1])) # Standard deviation of error on test set vs. option price with noise
                BS_mean_error.append(np.average(y_test[:,0] - y_test[:,1])) # Mean error on test set vs. BS analytical formula price
                BS_std_error.append(np.std(y_test[:,0] - y_test[:,1]))# Standard deviation of error on 

            # Naming conventions
            model_tags = ["Base", "Add 1 Hidden Layers", "Add 2 Hidden Layers", "Reduced Neurons", "Increased Neurons"]
            metrics_tags = ["MAE test values", "Mean Error", "Std error", "Mean error vs BS price", "Std error vs BS price", "BS mean error", "BS std error"]

            # Map variables
            res_ = list(zip(mae_test_vals, mean_error, std_error, mean_error_vs_BS_price, std_error_vs_BS_price, BS_mean_error, BS_std_error))
            results = pd.DataFrame(res_, index=model_tags, columns = metrics_tags)

            # Print results
            st.write("MAE for testing set for each case")
            st.dataframe(results)

    elif len(y_choice) != 2:
        st.error("Please choose 2 target variables: Dirty and Clean Option")
        