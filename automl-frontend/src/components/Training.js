import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';

const Training = ({ runId }) => {
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [modelsStatus, setModelsStatus] = useState([]);
  const [currentStage, setCurrentStage] = useState('Data Preprocessing');
  const navigate = useNavigate();

  const stages = [
    'Data Preprocessing',
    'Feature Engineering',
    'Model Training',
    'Hyperparameter Tuning',
    'Model Evaluation',
    'Finalizing Results'
  ];

  // Map backend model results to frontend progress objects
  const updateModelsFromResults = (rankedModels) => {
    if (!Array.isArray(rankedModels)) return;

    const updated = rankedModels.map((m, idx) => ({
      name: m.name || `Model ${idx + 1}`,
      status: 'done',
      progress: 100,
      accuracy: m.accuracy || m.r2 || null,
      score: m.suitability_score || null
    }));

    setModelsStatus(updated);
  };

  useEffect(() => {
    if (!runId) {
      navigate('/upload');
      return;
    }

    const pollBackend = setInterval(async () => {
      try {
        const response = await api.getResults(runId);
        const data = response.data;

        if (data.status === 'done' || data.status === 'error' || data.out?.status === 'done') {
          clearInterval(pollBackend);

          const resultData = data.out || data;
          setResults(resultData);

          // Update final models info
          updateModelsFromResults(resultData.ranked);
          setProgress(100);
          setCurrentStage('Finalizing Results');

          // Navigate after short delay
          setTimeout(() => navigate('/results'), 3000);
        } else {
          // Progress simulation while backend still training
          setProgress(prev => {
            const newProgress = Math.min(prev + Math.random() * 8, 95);
            const stageIndex = Math.floor(newProgress / (100 / stages.length));
            setCurrentStage(stages[stageIndex] || stages[0]);
            return newProgress;
          });

          // Optional: show dummy model progress until backend finishes
          setModelsStatus(prev => {
            if (prev.length === 0) {
              return [
                { name: 'Random Forest', progress: 30, status: 'training' },
                { name: 'XGBoost', progress: 20, status: 'training' },
                { name: 'Neural Network', progress: 10, status: 'training' },
                { name: 'LightGBM', progress: 25, status: 'training' }
              ];
            }
            return prev.map(m => ({
              ...m,
              progress: Math.min(m.progress + Math.random() * 5, 90)
            }));
          });
        }
      } catch (err) {
        console.error('Error polling backend:', err);
      }
    }, 3000);

    return () => clearInterval(pollBackend);
  }, [runId, navigate]);

  return (
    <section className="training-section">
      <div className="container">
        <h1>Training in Progress</h1>
        <p>Our AI is automatically analyzing your dataset and training optimal models...</p>

        <div className="training-content">
          <h2>AutoML Training Pipeline</h2>
          <p><strong>Current Stage:</strong> {currentStage}</p>

          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }}></div>
          </div>

          <p><strong>{Math.floor(progress)}% Complete</strong></p>

          <div className="pipeline-steps">
            {stages.map((stage) => (
              <div
                key={stage}
                className={`step ${stage === currentStage ? 'active' : ''}`}
              >
                {stage}
              </div>
            ))}
          </div>

          <div className="model-progress">
            <h3>Model Training Status</h3>
            <p>Live updates from your AutoML backend</p>

            {modelsStatus.length === 0 ? (
              <p>Loading models...</p>
            ) : (
              modelsStatus.map((model, index) => (
                <div key={index} className="model-item">
                  <span>{model.name}</span>
                  <div className="model-progress-bar">
                    <div
                      className={`model-progress-fill ${model.status}`}
                      style={{
                        width: `${model.progress}%`,
                        background:
                          model.status === 'done'
                            ? '#4caf50'
                            : model.status === 'training'
                            ? '#ff9800'
                            : '#f44336'
                      }}
                    ></div>
                  </div>
                  <span>
                    {model.status === 'done'
                      ? '✅'
                      : model.status === 'training'
                      ? `${Math.round(model.progress)}%`
                      : '❌'}
                  </span>
                  {model.accuracy && (
                    <span style={{ marginLeft: '10px' }}>
                      Acc: {(model.accuracy * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              ))
            )}
          </div>

          {progress === 100 && (
            <div style={{ textAlign: 'center', marginTop: '2rem' }}>
              <h3>🎉 Training Complete!</h3>
              {results?.ranked?.length > 0 && (
                <p>
                  Best model: <strong>{results.ranked[0].name}</strong> with score{' '}
                  <strong>{results.ranked[0].suitability_score?.toFixed(3)}</strong>
                </p>
              )}
              <button
                className="btn btn-primary"
                onClick={() => navigate('/results')}
              >
                View Results & Deploy
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default Training;