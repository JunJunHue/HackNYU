import React from "react";
import { Link } from "react-router-dom";
import "../styles/Home.css";

const Home = () => {
  return (
    <div className="home-container">
      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
          <h1>Smart Eco Landscaper</h1>
          <p>Smart eco-friendly landscaping solutions for a greener tomorrow.</p>
          <Link to="/dashboard">
            <button className="animated-button">
              <svg viewBox="0 0 24 24" className="arr-2" xmlns="http://www.w3.org/2000/svg">
                <path
                  d="M16.1716 10.9999L10.8076 5.63589L12.2218 4.22168L20 11.9999L12.2218 19.778L10.8076 18.3638L16.1716 12.9999H4V10.9999H16.1716Z"
                ></path>
              </svg>
              <span className="text">Get Started</span>
              <span className="circle"></span>
              <svg viewBox="0 0 24 24" className="arr-1" xmlns="http://www.w3.org/2000/svg">
                <path
                  d="M16.1716 10.9999L10.8076 5.63589L12.2218 4.22168L20 11.9999L12.2218 19.778L10.8076 18.3638L16.1716 12.9999H4V10.9999H16.1716Z"
                ></path>
              </svg>
            </button>
          </Link>
        </div>
      </div>

      {/* Features Section */}
      <div className="features-section">
        <h2>Why Choose Smart Eco Landscaper?</h2>
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