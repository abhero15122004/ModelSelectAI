import React from 'react';

const ModelCard = ({ model, rank, task }) => {
  const getMetricDisplay = () => {
    if (task === 'classification') {
      return (
        <div className="metrics-grid">
          <div className="metric">
            <div className="metric-label">Accuracy</div>
            <div className="metric-value">
              {model.accuracy ? model.accuracy.toFixed(4) : 'N/A'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">F1 Score</div>
            <div className="metric-value">
              {model.f1_weighted ? model.f1_weighted.toFixed(4) : 'N/A'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">ROC AUC</div>
            <div className="metric-value">
              {model.roc_auc ? model.roc_auc.toFixed(4) : 'N/A'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">Precision</div>
            <div className="metric-value">
              {model.precision ? model.precision.toFixed(4) : 'N/A'}
            </div>
          </div>
        </div>
      );
    } else if (task === 'regression') {
      return (
        <div className="metrics-grid">
          <div className="metric">
            <div className="metric-label">R² Score</div>
            <div className="metric-value">
              {model.r2 ? model.r2.toFixed(4) : 'N/A'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">MAE</div>
            <div className="metric-value">
              {model.mae ? model.mae.toFixed(4) : 'N/A'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">RMSE</div>
            <div className="metric-value">
              {model.rmse ? model.rmse.toFixed(4) : 'N/A'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">Suitability</div>
            <div className="metric-value">
              {model.suitability_score ? model.suitability_score.toFixed(4) : 'N/A'}
            </div>
          </div>
        </div>
      );
    } else {
      // For image/medical tasks
      return (
        <div className="metrics-grid">
          <div className="metric">
            <div className="metric-label">Accuracy</div>
            <div className="metric-value">
              {model.accuracy ? model.accuracy.toFixed(4) : 'N/A'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">Training Time</div>
            <div className="metric-value">
              {model.train_time_s ? model.train_time_s.toFixed(2) + 's' : 'N/A'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">Model Size</div>
            <div className="metric-value">
              {model.size_mb ? model.size_mb.toFixed(2) + 'MB' : 'N/A'}
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">Suitability</div>
            <div className="metric-value">
              {model.suitability_score ? model.suitability_score.toFixed(4) : 'N/A'}
            </div>
          </div>
        </div>
      );
    }
  };

  return (
    <div className="model-card">
      <div className="model-header">
        <h3>{model.name}</h3>
        <div className="model-rank">#{rank}</div>
      </div>
      
      {getMetricDisplay()}
      
      <div className="model-details">
        <p><strong>Training Time:</strong> {model.train_time_s ? model.train_time_s.toFixed(2) + 's' : 'N/A'}</p>
        <p><strong>Inference Time:</strong> {model.infer_time_s_per_row ? (model.infer_time_s_per_row * 1000).toFixed(2) + 'ms' : 'N/A'}</p>
        <p><strong>Model Size:</strong> {model.size_mb ? model.size_mb.toFixed(2) + 'MB' : 'N/A'}</p>
        <p><strong>Explainability:</strong> {model.explainability ? (model.explainability * 100).toFixed(0) + '%' : 'N/A'}</p>
      </div>
    </div>
  );
};

export default ModelCard;