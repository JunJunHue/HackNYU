import React, { useState } from "react";
import axios from "axios";
import "../styles/Dashboard.css";

const Dashboard = () => {
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [plants, setPlants] = useState([]);
  const [climate, setClimate] = useState([]);
  const [explanations, setExplanations] = useState({});
  const [error, setError] = useState("");

  // 📝 Function to Summarize AI Explanation
  const summarizeExplanation = (explanation) => {
    if (!explanation || explanation.trim().length === 0) {
      return "No explanation available.";
    }

    // Keep the first meaningful 2-3 sentences
    let sentences = explanation.split(". ").slice(0, 3).join(". ");
    
    // Ensure a clean end
    if (!sentences.endsWith(".")) sentences += ".";

    return sentences;
  };

  const fetchData = async () => {
    setError("");
    setPlants([]);
    setClimate([]);
    setExplanations({});

    if (!latitude || !longitude) {
      setError("Please enter valid latitude and longitude.");
      return;
    }

    try {
      console.log(`Fetching plant recommendations for lat=${latitude}, lon=${longitude}`);
      const plantResponse = await axios.get(`http://127.0.0.1:5000/predict?lat=${latitude}&lon=${longitude}`);

      let plantList = [];
      if (plantResponse.data && plantResponse.data.recommended_plants) {
        plantList = plantResponse.data.recommended_plants;
        setPlants(plantList);
      } else {
        setError("No plant recommendations found for this location.");
      }

      console.log("Fetching climate data...");
      const climateResponse = await axios.get(`http://127.0.0.1:5000/climate?lat=${latitude}&lon=${longitude}`);

      let climateData = [];
      if (climateResponse.data && climateResponse.data.climate_factors) {
        climateData = climateResponse.data.climate_factors;
        setClimate(climateData);
      } else {
        setError("No climate data available for this location.");
      }

      // Fetch AI explanations
      if (plantList.length > 0 && climateData.length > 0) {
        console.log("Fetching AI explanations...");
        const explanationResponse = await axios.post(
          `http://127.0.0.1:5000/explain`,
          { plants: plantList, climate: climateData },
          { headers: { "Content-Type": "application/json" } }
        );

        if (explanationResponse.data && explanationResponse.data.explanations) {
          console.log("🌱 AI Explanations Received:", explanationResponse.data.explanations);

          // Summarize explanations before storing
          const summarizedExplanations = {};
          Object.keys(explanationResponse.data.explanations).forEach((plant) => {
            summarizedExplanations[plant] = summarizeExplanation(explanationResponse.data.explanations[plant]);
          });

          setExplanations(summarizedExplanations);
        } else {
          console.error("❌ AI Explanation Error: No explanations found.");
        }
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      setError("Failed to fetch data. Please check your connection and try again.");
    }
  };

  const formatClimateData = (climate) => {
    const climateMapping = {
      RH2M: "Humidity (%)",
      WS2M: "Wind Speed (m/s)",
      T2MDEW: "Dew Point Temperature (°C)",
      ALLSKY_SFC_SW_DWN: "Actual Solar Radiation (MJ/m²/day)",
      T2M_MIN: "Temperature (Min) (°C)",
      QV2M: "Specific Humidity (g/kg)",
      T2M_MAX: "Temperature (Max) (°C)",
      PS: "Surface Pressure (kPa)",
      T2M: "Temperature (Avg) (°C)",
      ALLSKY_SFC_LW_DWN: "Longwave Downward Radiation (MJ/m²/day)",
      PRECTOTCORR: "Precipitation (mm/day)",
      CLRSKY_SFC_SW_DWN: "Theoretical Solar Radiation (MJ/m²/day)",
    };

    return climate.map(([key, value]) => {
      const label = climateMapping[key] || key;
      let formattedValue = value;

      if (key === "PRECTOTCORR") {
        formattedValue = value === 0 ? "No rainfall" : `${value} mm/day`;
      } else if (key === "RH2M") {
        formattedValue = `${value}%`;
      }

      return `${label} = ${formattedValue}`;
    });
  };

  return (
    <div className="dashboard-container">
      <h1>🌿 Find the Best Plants & Climate for Your Location</h1>

      <div className="search-section">
        <input
          type="text"
          placeholder="Enter latitude..."
          value={latitude}
          onChange={(e) => setLatitude(e.target.value)}
        />
        <input
          type="text"
          placeholder="Enter longitude..."
          value={longitude}
          onChange={(e) => setLongitude(e.target.value)}
        />
        <button className="dashboard-search-button" onClick={fetchData}>
          SEARCH
        </button>
      </div>

      {error && <p className="error-message">{error}</p>}

      {plants.length > 0 && (
        <div className="results-section">
          <h2>🌱 Recommended Plants:</h2>
          <div className="plant-list">
            {plants.map((plant, index) => (
              <div key={index} className="plant-card">
                <strong>{plant}</strong>
                {explanations[plant] ? (
                  <p className="explanation-text">{explanations[plant]}</p>
                ) : (
                  <p className="explanation-text">Loading Explanations...</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {climate.length > 0 && (
        <div className="results-section">
          <h2>🌤 Climate Factors:</h2>
          <div className="climate-list">
            {formatClimateData(climate).map((factor, index) => (
              <div key={index} className="climate-card">
                {factor}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;