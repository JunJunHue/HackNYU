import os
import torch
import torch.nn as nn
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai  # ✅ Google Gemini AI

# ✅ Load Google Gemini API Key (Ensure this is securely stored)
genai.configure(api_key="AIzaSyDouyI4CVYvKk-RS33daEmGggFycVJ6fzQ")

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# ✅ Define Neural Network Model
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
        x = self.sigmoid(self.fc3(x))  # Multi-label classification output
        return x

# ✅ Load Pretrained Model and Labels
try:
    species_labels = np.load("species_labels.npy", allow_pickle=True)
    num_species = len(species_labels)
    model = TreePredictor(input_size=15, output_size=num_species)
    model.load_state_dict(torch.load("tree_model.pth"))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ✅ Load Climate Data
def load_weather_data():
    try:
        with open("weatherOutput.json", "r") as f:
            data = json.load(f)
        print(f"✅ Successfully loaded weatherOutput.json with {len(data)} entries.")
        return data
    except Exception as e:
        print(f"❌ Error loading weatherOutput.json: {e}")
        return []

weather_data = load_weather_data()

# ✅ Initialize and Fit StandardScaler
scaler = StandardScaler()

def fit_scaler(data):
    X = []
    for entry in data:
        lat, lon = entry[0], entry[1]
        factors = [factor[1] for factor in entry[2:15]]  # Extract 13 factors
        X.append([lat, lon] + factors)

    if X:
        X = np.array(X)
        scaler.fit(X)
        print(f"✅ StandardScaler fitted with {len(X)} data points.")

fit_scaler(weather_data)

# ✅ Predict Tree Species
def predict_trees(lat, lon):
    """Predicts suitable trees based on location and climate data."""
    factors = get_factors(lat, lon, weather_data)
    if factors is None:
        return {"error": "Location not found in dataset."}

    full_input = np.array([lat, lon] + factors).reshape(1, -1)

    try:
        factors_scaled = scaler.transform(full_input)
        factors_tensor = torch.tensor(factors_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(factors_tensor)

        probabilities = output.numpy().flatten()
        top_indices = np.argsort(probabilities)[::-1][:5]
        top_species = [species_labels[idx] for idx in top_indices]

        return {"recommended_plants": top_species[:3]}

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

# ✅ Extract Climate Factors for a Given Location
def get_factors(lat, lon, data):
    """Retrieves climate factors for a given latitude and longitude."""
    for entry in data:
        if entry[0] == lat and entry[1] == lon:
            return [factor[1] for factor in entry[2:15]]
    return None  # Not found

# ✅ Use Gemini AI for Explanation Generation & Summarization
def generate_explanation(plant_name, climate_factors):
    """Generates and summarizes a plant explanation using Google Gemini AI."""
    try:
        climate_description = ", ".join([f"{key}: {value}" for key, value in climate_factors])
        prompt = f"Why is {plant_name} suitable for a climate with {climate_description}? Summarize the answer in 2-3 sentences."

        print(f"🤖 Sending prompt to Gemini: {prompt}")

        # Generate Explanation Using Gemini
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "No valid explanation generated."

    except Exception as e:
        print(f"❌ Error generating explanation for {plant_name}: {e}")
        return "Could not generate explanation."

# ✅ API Endpoint: Explain Plant Suitability
@app.route("/explain", methods=["POST"])
def explain():
    """API Endpoint to generate AI-based explanations."""
    try:
        data = request.json
        plants = data.get("plants", [])
        climate_factors = data.get("climate", [])

        if not plants or not climate_factors:
            return jsonify({"error": "Plants and climate data are required"}), 400

        explanations = {plant: generate_explanation(plant, climate_factors) for plant in plants}

        return jsonify({"explanations": explanations})
    
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# ✅ API Endpoint: Predict Plants
@app.route("/predict", methods=["GET"])
def predict():
    """API Endpoint to predict plants based on location."""
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude are required."}), 400

    result = predict_trees(lat, lon)
    return jsonify(result)

# ✅ API Endpoint: Retrieve Climate Data
@app.route("/climate", methods=["GET"])
def climate():
    """API Endpoint to retrieve climate data for a given location."""
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude are required."}), 400

    result = get_climate_data(lat, lon)
    
    if result is None:
        return jsonify({"error": "No climate data available for this location."}), 404

    return jsonify({"latitude": lat, "longitude": lon, "climate_factors": result})

# ✅ Retrieve Climate Data for a Location
def get_climate_data(lat, lon):
    """Finds climate data for a given latitude and longitude."""
    closest_entry = None
    min_diff = float("inf")

    for entry in weather_data:
        entry_lat, entry_lon = entry[0], entry[1]
        diff = abs(entry_lat - lat) + abs(entry_lon - lon)

        if diff < min_diff:
            min_diff = diff
            closest_entry = entry

    if closest_entry:
        cleaned_factors = [factor for factor in closest_entry[2:14] if factor[1] != -999.0]
        return cleaned_factors

    return None

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)