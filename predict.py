import pandas as pd


def create_target_variable(df, window_size=30, co2_threshold=800):
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
