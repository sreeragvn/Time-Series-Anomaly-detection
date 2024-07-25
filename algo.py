import pandas as pd
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.seasonal import seasonal_decompose
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import json

TIME_STEPS = 30
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 0.01
LATENT_DIM = 4
EARLY_STOPPING_PATIENCE = 20
HYPER_PARAMETER_TUNE = False
# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()

def create_dataset(X, time_steps=1):
    Xs = []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
    return np.array(Xs)

def split_train_test_sequence(df, TIME_STEPS):
    df = df.copy()
    df = df.reset_index()
    df['month'] = df['datetime'].dt.to_period('M')
    months = df['month'].unique()
    df = df.set_index("datetime")
    sequences = []

    for month in months:
        df_month = df[df["month"]==month]
        df_month = df_month.drop(columns=["month"])
        sequence = create_dataset(df_month, TIME_STEPS)
        sequences.append(sequence)
    return np.concatenate(sequences, axis=0)

def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

class DenseEncoder(nn.Module):
    def __init__(self, input_shape: int, latent_dim: int):
        super().__init__()
        self.l1 = nn.Linear(in_features=input_shape, out_features=4 * latent_dim)
        self.l2 = nn.Linear(in_features=4 * latent_dim, out_features=2 * latent_dim)
        self.l3 = nn.Linear(in_features=2 * latent_dim, out_features=latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        latent = torch.relu(x)
        return latent


class DenseDecoder(nn.Module):
    def __init__(self, output_shape: int, latent_dim: int):
        super().__init__()
        self.l4 = nn.Linear(in_features=latent_dim, out_features=2 * latent_dim)
        self.l5 = nn.Linear(in_features=2 * latent_dim, out_features=4 * latent_dim)
        self.output = nn.Linear(in_features=4 * latent_dim, out_features=output_shape)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.l4(latent)
        x = torch.relu(x)
        x = self.l5(x)
        x = torch.relu(x)
        output = self.output(x)

        return output


class DenseAutoencoderModel(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(DenseAutoencoderModel, self).__init__()
        self.encoder = DenseEncoder(input_shape, latent_dim)
        self.decoder = DenseDecoder(input_shape, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, inputs):
        inputs = inputs.squeeze(2)
        latent = self.encoder(inputs)
        output = self.decoder(latent)
        output = output.unsqueeze(2)
        return output

def train_model(model, train_loader, criterion, optimizer, epochs, patience):
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for df in train_loader:
            df = df[0].to(device)
            optimizer.zero_grad()
            output = model(df)
            loss = criterion(output, df)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {train_loss}')

        if train_loss < best_loss:
            best_loss = train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print('Early stopping!')
            break

    return best_loss

def objective(trial, train_loader):
    latent_dim = trial.suggest_int('latent_dim', 2, 6)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    model = DenseAutoencoderModel(input_shape=TIME_STEPS, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    final_loss = train_model(model, train_loader, criterion, optimizer, EPOCHS, EARLY_STOPPING_PATIENCE)
    
    return final_loss

def run_optuna(train_loader):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader), n_trials=50)
    best_params = study.best_params
    print(f'Best hyperparameters: {best_params}')
    return best_params


def preprocess_and_train(train, column_name):
    train_col = pd.DataFrame(train, columns=[column_name])

    TIME_STEPS = 30
    BATCH_SIZE = 512
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20

    # split_train_test_sequence
    X_train = create_dataset(train_col, TIME_STEPS)

    min_val = X_train.min()
    max_val = X_train.max()  # Use train max for consistency

    train_data = normalize_data(X_train, min_val, max_val)

    train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    if HYPER_PARAMETER_TUNE:
        best_params = run_optuna(train_loader)

        with open(f'saves/best_hyperparameters_{column_name}.json', 'w') as f:
            json.dump(best_params, f)

        model = DenseAutoencoderModel(input_shape=TIME_STEPS, latent_dim=best_params['latent_dim']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        
    else:
        with open(f'saves/best_hyperparameters_{column_name}.json', 'r') as f:
            best_params = json.load(f)
        model = DenseAutoencoderModel(input_shape=TIME_STEPS, latent_dim=best_params['latent_dim']).to(device)
        # model.load_state_dict(torch.load('saves/best_autoencoder_model_data_0.pth'))
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    final_loss = train_model(model, train_loader, criterion, optimizer, EPOCHS, EARLY_STOPPING_PATIENCE)
    print(f'Final model training loss for {column_name}: {final_loss}')

    if HYPER_PARAMETER_TUNE:
        torch.save(model.state_dict(), f'saves/best_autoencoder_model_{column_name}.pth')
        print(f'Best model for {column_name} saved!')

    return model, min_val, max_val, best_params

def calculate_anomalies(data_org, column_name, model, min_val, max_val, threshold):
    # Create a windowed dataset for the specified column
    data_window = create_dataset(data_org[[column_name]], TIME_STEPS)
    
    # Scale the data
    data_window_scale = (data_window - min_val) / (max_val - min_val)
    
    # Convert to PyTorch tensor
    data_window_scale = torch.tensor(data_window_scale, dtype=torch.float32).to(device)
    
    # Create a DataLoader
    data_loader = torch.utils.data.DataLoader(data_window_scale, batch_size=1, shuffle=False)
    
    # Calculate reconstruction losses
    reconstruction_loss = []
    with torch.no_grad():
        for df in data_loader:
            df = df.to(device)
            output = model(df)
            loss = criterion(output, df)
            reconstruction_loss.append(loss.item())
    
    # Convert to numpy array
    array_of_values = np.array(reconstruction_loss)
    
    # Identify anomalies
    is_anomaly = array_of_values > threshold
    
    # Prepare column name for anomaly flag
    anomaly_column = f"is_anomaly_{column_name.split('_')[1]}"
    data_org[anomaly_column] = False
    
    # Calculate the starting index for updating the original DataFrame
    n = len(is_anomaly)
    start_idx = -(n + 5)
    if start_idx < 0:
        start_idx = max(len(data_org) + start_idx, 0)
    
    # Get the rows to update
    rows_to_update = data_org.index[start_idx:start_idx + n]
    
    # Update the DataFrame with anomaly information
    data_org.loc[rows_to_update, anomaly_column] = is_anomaly
    
    return reconstruction_loss

def sensor_difference_threshold_compute(df):
    df = df.copy()
    # Standard Deviation Method
    mu = df['diff'].mean()
    sigma = df['diff'].std()
    k = 3  # Typically 2 or 3
    threshold_std_max = mu + k * sigma
    threshold_std_min = mu - k * sigma

    threshold_std_min = min(-threshold_std_max, threshold_std_min)
    threshold_std_max = max(threshold_std_max, -threshold_std_min)

    # Percentile Method
    threshold_perc = np.percentile(np.abs(df['diff']), 99)

    # Interquartile Range (IQR) Method
    Q1 = df['diff'].quantile(0.25)
    Q3 = df['diff'].quantile(0.75)
    IQR = Q3 - Q1
    k = 1.5
    threshold_iqr = Q3 + k * IQR

    # print(f"Threshold using IQR method: {threshold_iqr}")

    #since the df is normally distributed
    print(f"Threshold using standard deviation method: {threshold_std_max, threshold_std_min}")

    # if the df is not normally distributed
    # print(f"Threshold using 99th percentile method: {threshold_perc}")

    # Moving average threshold
    # window_size = 20
    # df['rolling_mean'] = df['diff'].rolling(window=window_size).mean()
    # df['rolling_std'] = df['diff'].rolling(window=window_size).std()
    # df['threshold_moving'] = df['rolling_mean'] + 2 * df['rolling_std']
    # print("Moving average threshold is also calculated")

    threshold_max = threshold_std_max
    threshold_min = threshold_std_min
    df['is_anomaly'] = (df['diff'] > threshold_max) | (df['diff'] < threshold_min)
    return df

def remove_anomaly_and_fill(data):
    data = data.copy()
    if "is_anomaly" in data.columns:
        data = data[data["is_anomaly"] == False]
    elif "is_anomaly_0" in data.columns and "is_anomaly_1" in data.columns:
        data = data[(data["is_anomaly_0"] == False) & (data["is_anomaly_1"] == False)]
    else:
        raise ValueError("Required anomaly columns are missing in the data.")
    data = data.resample('5T').ffill()
    data = data[["data_0", "data_1"]]
    return data

def univariate_GMM_anomaly_detection(data):
    gmm_0 = GaussianMixture(n_components=3)
    gmm_0.fit(data[['data_0']])

    gmm_1 = GaussianMixture(n_components=3)
    gmm_1.fit(data[['data_1']])

    # Anomaly detection
    data['is_anomaly_0'] = gmm_0.score_samples(data[['data_0']])
    data['is_anomaly_1'] = gmm_1.score_samples(data[['data_1']])

    # Determine a threshold for anomalies (for example, lower 1% of scores)
    threshold_0 = np.percentile(data['is_anomaly_0'], 0.55)
    threshold_1 = np.percentile(data['is_anomaly_1'], 0.55)

    data['is_anomaly_0'] = data['is_anomaly_0'] < threshold_0
    data['is_anomaly_1'] = data['is_anomaly_1'] < threshold_1
    return data

def multivariate_GMM_anomaly_detection(data):
    sensor1_data = data['data_0'].values
    sensor2_data = data['data_1'].values

    # Combine the two sensor data into a single 2D array
    combined_data = np.vstack((sensor1_data, sensor2_data)).T

    # Define the number of components (clusters) for the GMM
    n_components = 3  # Since you mentioned the distribution is bimodal

    # Fit a 2D GMM to the combined data
    gmm_2d = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm_2d.fit(combined_data)

    # Get the parameters of the fitted GMM
    means = gmm_2d.means_
    covariances = gmm_2d.covariances_
    weights = gmm_2d.weights_

    print("Means:\n", means)
    print("Covariances:\n", covariances)
    print("Weights:\n", weights)

    # Calculate the log-likelihood of each point
    log_likelihood = gmm_2d.score_samples(combined_data)

    # Set a threshold for anomaly detection
    threshold = np.percentile(log_likelihood, 0.10)  # For example, flagging the lowest 1% as anomalies

    # Identify anomalies
    data['anomaly_score'] = log_likelihood
    data['is_anomaly'] = data['anomaly_score'] < threshold
    return data

def decompose_and_detect_anomalies(data_series, period, k):
    decomposition = seasonal_decompose(data_series, model='additive', period=period)
    residual = decomposition.resid

    threshold_upper = residual.mean() + k * residual.std()
    threshold_lower = residual.mean() - k * residual.std()

    is_anomaly = (residual > threshold_upper) | (residual < threshold_lower)

    return residual, decomposition, is_anomaly

def timeseries_decomposition_anomaly_threshold(data, period=288, k=3):
    data = data.copy()
    data = data.asfreq('5min')

    # Decompose and detect anomalies for data_0
    residual_0, decomposition_0, is_anomaly_0 = decompose_and_detect_anomalies(data['data_0'], period, k)
    data['is_anomaly_0'] = is_anomaly_0

    # Decompose and detect anomalies for data_1
    residual_1, decomposition_1, is_anomaly_1 = decompose_and_detect_anomalies(data['data_1'], period, k)
    data['is_anomaly_1'] = is_anomaly_1

    return residual_0, residual_1, decomposition_0, decomposition_1, data
