import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import ModelCard from './ModelCard';

const Results = ({ runId, uploadData }) => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    if (!runId) {
      navigate('/upload');
      return;
    }

    const fetchResults = async () => {
      try {
        const response = await api.getResults(runId);
        const data = response.data;
        
        if (data.status === 'done') {
          setResults(data);
        } else if (data.status === 'running') {
          // If still running, check again in 2 seconds
          setTimeout(fetchResults, 2000);
          return;
        }
      } catch (error) {
        console.error('Error fetching results:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [runId, navigate]);

  if (loading) {
    return (
      <section className="results-section">
        <div className="container">
          <div className="loading">
            <div className="spinner"></div>
            <p>Loading results...</p>
          </div>
        </div>
      </section>
    );
  }

  if (!results) {
    return (
      <section className="results-section">
        <div className="container">
          <h1>Results Not Available</h1>
          <p>Unable to load results. Please try again.</p>
        </div>
      </section>
    );
  }

  const { task, top5, figs = [], graphs = [], errors = [] } = results;

  return (
    <section className="results-section">
      <div className="container">
        <h1>Model Training Results</h1>
        <p>Application: {uploadData?.application || 'generic'} | Task: {task}</p>

        <div className="top-models">
          {top5.map((model, index) => (
            <ModelCard 
              key={model.name}
              model={model}
              rank={index + 1}
              task={task}
            />
          ))}
        </div>

        {(figs.length > 0 || graphs.length > 0) && (
          <div className="visualizations">
            <h2>Visualizations</h2>
            
            {figs.map((fig, index) => (
              <div key={index} className="viz-card">
                <h3>Figure {index + 1}</h3>
                <img 
                  src={api.getFigure(runId, fig)} 
                  alt={`Training result ${index + 1}`}
                  className="viz-image"
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
              </div>
            ))}

            {graphs.map((graph, index) => (
              <div key={index} className="viz-card">
                <h3>Graph {index + 1}</h3>
                <img 
                  src={api.getGraph(runId, graph)} 
                  alt={`Analysis graph ${index + 1}`}
                  className="viz-image"
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
              </div>
            ))}
          </div>
        )}

        {errors.length > 0 && (
          <div className="errors-section">
            <h2>Errors</h2>
            <div className="errors-list">
              {errors.map((error, index) => (
                <div key={index} className="error-item">
                  {error}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default Results;