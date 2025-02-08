import React from "react";
import { Link } from "react-router-dom";
import "../styles/Home.css";

const Home = () => {
  return (
    <div className="home-container">
      <div className="content">
        <h1>Smart Eco Landscaping</h1>
        <p>Design sustainable, climate-resilient landscapes effortlessly.</p>
        <Link to="/dashboard">
          <button>Get Started</button>
        </Link>
      </div>
    </div>
  );
};

export default Home;