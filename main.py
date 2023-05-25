import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense


def create_target_variable(df, window_size=30, co2_threshold=600):
    """
    Create a target variable (in_room) based on the CO2 values and their rolling window statistics.

    Parameters:
    df (pd.DataFrame): A pandas dataframe with two columns: 'timestamp' and 'co2'.
    window_size (int): The number of periods to include in the rolling window.
    co2_threshold (float): CO2 value above which it is considered that someone is in the room.

    Returns:
    pd.DataFrame: A new dataframe with an additional 'in_room' column indicating if someone is in the room.
    """
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy['timestamp'])

    # Calculate the rolling mean, median, and standard deviation
    df_copy['rolling_mean'] = df_copy['co2'].rolling(window=window_size, min_periods=1).mean()
    df_copy['rolling_median'] = df_copy['co2'].rolling(window=window_size, min_periods=1).median()
    df_copy['rolling_std'] = df_copy['co2'].rolling(window=window_size, min_periods=1).std().fillna(0)

    # Create a custom condition to determine if someone is in the room
    condition = (df_copy['co2'] > co2_threshold) & (df_copy['rolling_mean'] > co2_threshold) & (
                df_copy['rolling_median'] > co2_threshold)

    df_copy['in_room'] = condition.astype(int)

    return df_copy


def preprocess_lstm_data(df, sequence_length=24*60):
    df_copy = create_target_variable(df)
    data = df_copy['in_room'].values

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])

    if len(X) == 0:  # If no sequence was created, create one without a target
        X.append(data[:sequence_length])
        y.append(np.nan)

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y


def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def lstm_predict_and_save(df, sequence_length=24*60, epochs=10, batch_size=64, train_fraction=0.8, model_path='lstm_model.h5'):
    X, y = preprocess_lstm_data(df, sequence_length)
    train_size = int(train_fraction * len(X))
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, shuffle=False)

    # Save the trained model
    model.save(model_path)

    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).astype(int)

    # Add timestamps to the predictions
    timestamp_index = pd.to_datetime(df['timestamp']).iloc[sequence_length + train_size:].reset_index(drop=True)
    predictions = pd.DataFrame({'timestamp': timestamp_index, 'in_room': y_pred.flatten()})

    return predictions


def predict_with_saved_model(new_df, sequence_length=24*60, model_path='lstm_model.h5'):
    X_new, y_new = preprocess_lstm_data(new_df, sequence_length)
    #X_new, _ = preprocess_lstm_data(new_df, sequence_length)

    # Load the saved model
    loaded_model = load_model(model_path)

    # Make predictions on new data
    y_pred_new = loaded_model.predict(X_new)
    y_pred_new = np.round(y_pred_new).astype(int)

    if np.isnan(y_new).all():  # If target is NaN, just use the end of the dataframe for timestamp
        timestamp_index_new = new_df['timestamp'].iloc[-1:]
    else:
        timestamp_index_new = pd.to_datetime(new_df['timestamp']).iloc[sequence_length:].reset_index(drop=True)

    # Add timestamps to the predictions
    #timestamp_index_new = pd.to_datetime(new_df['timestamp']).iloc[sequence_length:].reset_index(drop=True)
    predictions_new = pd.DataFrame({'timestamp': timestamp_index_new, 'in_room': y_pred_new.flatten()})

    return predictions_new


def main(df_train, df_new, sequence_length=24*60, epochs=10, batch_size=64, train_fraction=0.8, model_path='lstm_model.h5'):
    print("Training and saving the LSTM model...")
    predictions_train = lstm_predict_and_save(df_train, sequence_length, epochs, batch_size, train_fraction, model_path)

    print("Making predictions on new data using the saved model...")
    predictions_new = predict_with_saved_model(df_new, sequence_length, model_path)

    return predictions_train, predictions_new


#31 Augustus 2020 tot en met 30 April 2023 van sensor 3291 in room 0.30
#data_train = pd.read_csv('C:/Users/Patrick Ten brinke/Downloads/1598824800-1682891999-datapoint_3291.csv', sep=",")
#data_train.rename(columns = {'value':'co2'}, inplace = True)

#Maandag 8 Mei
data_new = pd.read_csv('C:/Users/Patrick Ten brinke/Downloads/1683583200-1683755999-datapoint_3291.csv', sep=",")
print(data_new.describe())
print('Row count is:', len(data_new.index))


#data_new = pd.read_csv('C:/Users/Patrick Ten brinke/Downloads/1683496800-1683755999-datapoint_3291.csv', sep=",")
data_new.rename(columns = {'value':'co2'}, inplace = True)
data_new_copy = data_new.copy()
#data_new_copy.drop(columns=['co2'])

#Assuming you have two dataframes named 'data_train' and 'data_new'
#predictions_train, predictions_new = main(data_train, data_new)

predictions_new = predict_with_saved_model(data_new_copy)

print('hoi', predictions_new)

