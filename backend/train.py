import json
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load JSON data
def load_data(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

# Process dataset
def prepare_data(data):
    X, Y = [], []
    all_species = set()
    
    for entry in data:

        lat, lon = entry[0], entry[1]
        factors = [factor[1] for factor in entry[2:15]]  # Extract 13 environmental factors
        species_list = entry[15]  # Extract plant species
        
        X.append([lat, lon] + factors)  # Include lat/lon as features
        species_counts = {species[0]: species[1] for species in species_list}
        
        sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
        top_species = [species[0] for species in sorted_species[:3]]  # Select top 3 species
        all_species.update(top_species)
        
        Y.append(top_species)
    
    return np.array(X), Y, list(all_species)

# Load and preprocess data
data = load_data("weatherOutput.json")
X_data, Y_data, unique_species = prepare_data(data)

# Encode species
species_to_index = {species: i for i, species in enumerate(unique_species)}
num_species = len(species_to_index)

def encode_labels(Y_data):

    Y_encoded = np.zeros((len(Y_data), num_species))
    for i, species_list in enumerate(Y_data):
        for species in species_list:
            if species in species_to_index:
                Y_encoded[i, species_to_index[species]] = 1  # One-hot encoding
    return Y_encoded

Y_data = encode_labels(Y_data)

# Normalize features
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

# Convert to PyTorch Tensors
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
Y_train, Y_test = torch.tensor(Y_train, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train, Y_train)

test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define Neural Network
class TreePredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(TreePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)

        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Multi-label output
        return x

# Initialize Model
input_size = X_data.shape[1]

output_size = num_species
model = TreePredictor(input_size, output_size)
criterion = nn.BCELoss()  # Binary cross-entropy for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 100
for epoch in range(epochs):
    model.train()

    total_loss = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "tree_model.pth")


np.save("species_labels.npy", unique_species)
print("Model training complete and saved.")
