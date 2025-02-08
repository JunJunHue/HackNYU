import React, { useState } from "react";
import axios from "axios";

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
    <div>
      <h1>Find the Best Plants for Your Area</h1>
      <input
        type="text"
        placeholder="Enter location..."
        value={location}
        onChange={(e) => setLocation(e.target.value)}
      />
      <button onClick={fetchPlants}>Get Recommendations</button>

      <ul>
        {plants.map((plant, index) => (
          <li key={index}>{plant.name} - {plant.description}</li>
        ))}
      </ul>
    </div>
  );
};

export default Dashboard;