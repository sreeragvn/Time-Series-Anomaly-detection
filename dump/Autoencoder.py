import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import optuna
import json

TIME_STEPS = 30
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 0.01
LATENT_DIM = 4
EARLY_STOPPING_PATIENCE = 20

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data_org = pd.read_csv("times_series_data_no_labels.csv", index_col='datetime', parse_dates=['datetime'])

def preprocess_data(column_name):
    # Removing rows with values greater than 32 and less than 19
    data = data_org[(data_org[column_name] <= 32) & (data_org[column_name] >= 19)]

    # Removing rows for the time between 5:45 and 21:00 with values less than 26
    data['hour'] = data.index.hour
    data['minute'] = data.index.minute

    condition_time = ~((data['hour'] > 5) & ((data['hour'] < 21) | ((data['hour'] == 21) & (data['minute'] == 0))) & (data[column_name] < 26))
    data = data[condition_time]

    # Dropping the additional columns used for filtering
    data.drop(columns=['hour', 'minute'], inplace=True)

    # Removing rows for the time between 00:10 and 03:05 with values greater than 22.5
    data['hour'] = data.index.hour
    data['minute'] = data.index.minute

    condition_night = ~((data['hour'] == 0) & (data['minute'] >= 10) |
                        (data['hour'] > 0) & (data['hour'] < 3) |
                        (data['hour'] == 3) & (data['minute'] <= 5) &
                        (data[column_name] > 22.5))
    data = data[condition_night]

    # Dropping the additional columns used for filtering
    data.drop(columns=['hour', 'minute'], inplace=True)

    # Split data into training and test sets
    train_size = int(len(data) * 0.85)
    train, test = data.iloc[0:train_size], data.iloc[train_size:len(data)]

    return train, test

train_0, test_0 = preprocess_data('data_0')
train_1, test_1 = preprocess_data('data_1')

def create_dataset(X, time_steps=1):
    Xs = []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
    return np.array(Xs)

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
    latent_dim = trial.suggest_int('latent_dim', 2, 12)
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

    X_train = create_dataset(train_col, TIME_STEPS)

    min_val = X_train.min()
    max_val = X_train.max()  # Use train max for consistency

    train_data = normalize_data(X_train, min_val, max_val)

    train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    best_params = run_optuna(train_loader)

    with open(f'best_hyperparameters_{column_name}.json', 'w') as f:
        json.dump(best_params, f)

    model = DenseAutoencoderModel(input_shape=TIME_STEPS, latent_dim=best_params['latent_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    final_loss = train_model(model, train_loader, criterion, optimizer, EPOCHS, EARLY_STOPPING_PATIENCE)
    print(f'Final model training loss for {column_name}: {final_loss}')

    torch.save(model.state_dict(), f'best_autoencoder_model_{column_name}.pth')
    print(f'Best model for {column_name} saved!')

    return model, min_val, max_val, best_params

criterion = nn.L1Loss()

# Train models for data_0 and data_1
model_0, min_val_0, max_val_0, best_params_0 = preprocess_and_train(train_0, 'data_0')
model_1, min_val_1, max_val_1, best_params_1 = preprocess_and_train(train_1, 'data_1')

def calculate_anomalies(column_name, model, min_val, max_val, threshold):
    data_window = create_dataset(data_org[[column_name]], TIME_STEPS)
    data_window_scale = (data_window - min_val) / (max_val - min_val)

    data_window_scale = torch.tensor(data_window_scale, dtype=torch.float32).to(device)
    data_loader = torch.utils.data.DataLoader(data_window_scale, batch_size=1, shuffle=False)

    reconstruction_loss = []
    with torch.no_grad():
        for df in data_loader:
            df = df.to(device)
            output = model(df)
            loss = criterion(output, df)
            reconstruction_loss.append(loss.item())

    array_of_values = np.array(reconstruction_loss)
    is_anomaly = array_of_values > threshold

    data_org[f"is_anomaly_{column_name}"] = False
    n = len(is_anomaly)
    start_idx = -(n + 5)

    if start_idx < 0:
        start_idx = max(len(data_org) + start_idx, 0)

    rows_to_update = data_org.index[start_idx:start_idx + n]
    data_org.loc[rows_to_update, f'is_anomaly_{column_name}'] = is_anomaly

    return reconstruction_loss

threshold_0 = 0.055  # Set threshold for data_0
threshold_1 = 0.055  # Set threshold for data_1

reconstruction_loss_0 = calculate_anomalies('data_0', model_0, min_val_0, max_val_0, threshold_0)
reconstruction_loss_1 = calculate_anomalies('data_1', model_1, min_val_1, max_val_1, threshold_1)

# Plot histograms
plt.hist(reconstruction_loss_0, bins=100)
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.title('Histogram of Reconstruction Losses for data_0')
plt.show()

plt.hist(reconstruction_loss_1, bins=100)
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.title('Histogram of Reconstruction Losses for data_1')
plt.show()

# Plot anomalies
from plot_anomaly import multivariate_anomaly_plot
multivariate_anomaly_plot(data=data_org)
