import React, { useState } from "react";
import axios from "axios";
import "../styles/Dashboard.css";

const Dashboard = () => {
  const [location, setLocation] = useState("");
  const [plants, setPlants] = useState([]);

  const fetchPlants = async () => {
    try {
      const response = await axios.get(
        `http://127.0.0.1:5000/recommend-plants?location=${location}`
      );
      setPlants(response.data);
    } catch (error) {
      console.error("Error fetching plant recommendations:", error);
    }
  };

  return (
    <div className="dashboard-container">
      <h1>🌿 Find the Best Plants for Your Area</h1>
      
      <div className="search-section">
        <input
          type="text"
          placeholder="Enter your location..."
          value={location}
          onChange={(e) => setLocation(e.target.value)}
        />
        <a href="#" onClick={fetchPlants}>
          <span>SEARCH</span>
        </a>
      </div>

      <div className="results-section">
        {plants.length > 0 ? (
          <div className="plant-list">
            {plants.map((plant, index) => (
              <div key={index} className="plant-card">
                <h3>{plant.name}</h3>
                <p>{plant.description}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="no-results">Enter a location to get recommendations.</p>
        )}
      </div>
    </div>
  );
};

export default Dashboard;