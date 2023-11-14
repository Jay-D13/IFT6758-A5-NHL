from comet_ml import Experiment
import argparse
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score
import torch
import torch.nn as nn
import pandas as pd
import os
from ift6758.training.net_model import Net
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

nn_configs = {
    'small_nn_1': [32, 16],
    'small_nn_2': [64],
    'small_nn_3': [64, 64],
    'small_nn_4': [256, 128],
    'deeper_nn_1': [256, 128, 64, 64, 32],
    'deeper_nn_2': [512, 256, 128, 64, 32, 16],
    }
nn_epochs = 15
nn_lr = 0.001
models_dict = {}

workspace = 'ift6758-a5-nhl'
project_name = 'milestone2'

def train_mlp(num_epochs, model, criterion, optimizer, train_loader, val_loader):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X = X.to(model.device)
            y = y.to(model.device)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        print(f'Epoch: {epoch + 1} \tTraining Loss: {train_loss:.6f}')
        val_loss, probabilities = val_mlp(model, criterion, val_loader)
        val_losses.append(val_loss)

    return train_losses, val_losses, probabilities

def val_mlp(model, criterion, val_loader):
    model.eval()
    val_loss = 0.0
    probabilities = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(model.device)
            y = y.to(model.device)

            outputs = model(X)
            loss = criterion(outputs, y.unsqueeze(1))

            val_loss += loss.item() * X.size(0)
            probabilities.extend(outputs.cpu().numpy()[:, 0])
            

    val_loss /= len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.6f}')

    return val_loss, probabilities

def run_nn_config(model_name, hidden_layers, input_shape, train_loader, val_loader, y_val, exp_folder):
    print(f'------ Training nn model: {model_name} ------')
    # Setup Experiment
    exp = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        workspace=workspace,
        project_name=project_name
    )
    exp.set_name(model_name)
    tags = ['CustomModels', 'NeuralNetwork', model_name,  'AllFeatures']
    exp.add_tags(tags)

    # Train model
    model = Net(input_shape, hidden_layers, 1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_lr)
    train_losses, val_losses, probabilities = train_mlp(nn_epochs, model, criterion, optimizer, train_loader, val_loader)
    models_dict[model_name] = probabilities 

    # Log to comet
    thresholds = np.arange(0.0, 1.0, 0.1)
    f1_scores = []; recall_scores = []; accuracy_scores = []
    for threshold in thresholds:
        prob_thresh = [1 if value > threshold else 0 for value in probabilities]
        f1_scores.append(f1_score(y_val, prob_thresh))
        recall_scores.append(recall_score(y_val, prob_thresh))
        accuracy_scores.append(accuracy_score(y_val, prob_thresh))
    
    for epoch in range(nn_epochs):
        exp.log_metric('Training loss', train_losses[epoch], step=epoch)
        exp.log_metric('Validation loss', val_losses[epoch], step=epoch)

    exp.log_curve('F1 score over different thresholds', x=thresholds, y=f1_scores)
    exp.log_curve('Recall score over different thresholds', x=thresholds, y=recall_scores)
    exp.log_curve('Accuracy score over different thresholds', x=thresholds, y=accuracy_scores)
    
    model_folder = os.path.join(exp_folder, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_path = os.path.join(model_folder, 'model.pt')
    torch.save(model.state_dict(), model_path)
    exp.log_model('Model', model_path)
    id = np.argmax(f1_scores)
    exp.log_metric('Accuracy', accuracy_scores[id])
    exp.log_metric('Recall', recall_scores[id])
    exp.log_metric('F1 score', f1_scores[id])

    exp.end()

def main(opts):
    # Create train folder
    exp_folder = os.path.join(opts.exp_path, opts.exp_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    # Get Data
    train_val = pd.read_pickle(opts.data_path)
    X_all = train_val.drop(['is_goal'], axis=1)
    y_all = train_val['is_goal']
    
    # Split into train val
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

    # Create dataloaders for NN
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Converting to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    # Create TensorDatasets for training and testing data
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    val_data = TensorDataset(X_val_tensor, y_val_tensor)

    # Create DataLoaders
    batch_size = 512
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Train NN models
    for (model_name, hidden_layers) in nn_configs.items():
        run_nn_config(model_name, hidden_layers, X_train.shape[1], train_loader, val_loader, y_val, exp_folder)

    
def parse_opts(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= './notebooks/Milestone2/TrainValSets2.pkl', help='Data path that contains train and val data')
    parser.add_argument('--exp_path', type=str, default= './train/', help='Experience path of parent folder')
    parser.add_argument('--exp_name', type=str, default= 'exp', help='Experience name for comet ml')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

def run(**kwargs):
    opts = parse_opts(True)
    for k, v in kwargs.items():
        setattr(opts, k, v)
    main(opts)
    return opts

if __name__ == '__main__':
    opts = parse_opts()
    main(opts)