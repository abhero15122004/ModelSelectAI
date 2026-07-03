import React from 'react';
import { Link } from 'react-router-dom';

const Hero = () => {
  return (
    <section className="hero">
      <div className="container">
        <div className="hero-content">
          <h1>Automated Machine Learning</h1>
          <p>Transform your data into intelligent models with zero coding required. Let AI build, optimize, and deploy machine learning models automatically.</p>
          
          <div className="cta-buttons">
            <Link to="/upload" className="btn btn-primary">
              Start Building Models →
            </Link>
            <a href="#demo" className="btn btn-secondary">
              View Demo
            </a>
          </div>

          <div className="stats">
            <div className="stat-item">
              <h3>Model Training</h3>
              <p>95% Accuracy</p>
            </div>
            <div className="stat-item">
              <h3>Auto Feature Engineering</h3>
              <p>Smart preprocessing</p>
            </div>
            <div className="stat-item">
              <h3>Hyperparameter Tuning</h3>
              <p>Optimized models</p>
            </div>
            <div className="stat-item">
              <h3>Model Selection</h3>
              <p>Best algorithms</p>
            </div>
            <div className="stat-item">
              <h3>Data Processed</h3>
              <p>1.2M Records</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;