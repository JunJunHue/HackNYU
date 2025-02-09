import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

# Load trained model
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

# Load model and species labels
species_labels = np.load("species_labels.npy", allow_pickle=True)
num_species = len(species_labels)
model = TreePredictor(input_size=15, output_size=num_species)  # 15 = lat, lon + 13 factors
model.load_state_dict(torch.load("tree_model.pth"))
model.eval()

# Load dataset to find environmental factors
def load_data(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

data = load_data("weatherOutput.json")
scaler = StandardScaler()

# Find environmental factors for a given coordinate
def get_factors(lat, lon, data):
    for entry in data:
        if entry[0] == lat and entry[1] == lon:
            return [factor[1] for factor in entry[2:15]]  # Extract 13 factors
    return None  # Not found

# Predict tree species for a given location
def predict_trees(lat, lon):
    factors = get_factors(lat, lon, data)
    if factors is None:
        return {"error": "Location not found in dataset."}
    
    # 🔥 FIX: Include lat & lon in input
    full_input = [lat, lon] + factors  

    # Normalize input
    factors_scaled = scaler.fit_transform([full_input])  # Normalize the full input
    factors_tensor = torch.tensor(factors_scaled, dtype=torch.float32)

    # Run prediction
    with torch.no_grad():
        output = model(factors_tensor)
    
    probabilities = output.numpy().flatten()
    top_indices = np.argsort(probabilities)[::-1][:3]  # Top 3 species
    top_species = [species_labels[idx] for idx in top_indices]
    
    return {"recommended_plants": top_species}

# Example usage
if __name__ == "__main__":
    lat, lon = 24.0, -112.0  # Example coordinates
    result = predict_trees(lat, lon)
    print(result)
