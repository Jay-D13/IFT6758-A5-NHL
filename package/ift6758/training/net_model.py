import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size=1):
        super(Net, self).__init__()
        
        sizes = [input_size] + layer_sizes + [output_size]
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:  # Hidden layers
                x = F.relu(layer(x))
                x = F.dropout(x, p=0.5, training=self.training)
            else:  # Output layer
                x = torch.sigmoid(layer(x))
        return x
    
    def get_pred_proba(self, X_test):
        self.eval()
        with torch.no_grad():
            # Assuming X_test is already a PyTorch tensor
            inputs = X_test.to(self.device)
            outputs = self(inputs)
            probabilities = outputs.cpu().numpy()[:, 1]
        return probabilities
