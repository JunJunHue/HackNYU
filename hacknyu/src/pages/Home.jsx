import React from "react";
import { Link } from "react-router-dom";
import "../styles/Home.css";

const Home = () => {
  return (
    <div className="home-container">
      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
          <h1>Transform Your Landscape with AI</h1>
          <p>Smart eco-friendly landscaping solutions for a greener tomorrow.</p>
          <Link to="/dashboard">
            <button className="btn-17">
              <span className="text-container">
                <span className="text">Get Started</span>
              </span>
            </button>
          </Link>
        </div>
      </div>

      {/* Features Section */}
      <div className="features-section">
        <h2>Why Choose Smart Eco Landscaping?</h2>
        <div className="features-container">
          <div className="feature-card">
            <img src="/images/leaf-icon.png" alt="Sustainability" />
            <h3>Eco-Friendly</h3>
            <p>Our AI recommends sustainable plants suited for your environment.</p>
          </div>
          <div className="feature-card">
            <img src="/images/water-icon.png" alt="Water Saving" />
            <h3>Water Efficient</h3>
            <p>Save water with smart irrigation suggestions and drought-resistant plants.</p>
          </div>
          <div className="feature-card">
            <img src="/images/shield-icon.png" alt="Wildfire Protection" />
            <h3>Fire-Resistant</h3>
            <p>Reduce wildfire risks with AI-powered fire-resistant landscaping tips.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;