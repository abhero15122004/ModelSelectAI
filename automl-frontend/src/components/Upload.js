import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';

const Upload = ({ onUploadComplete }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [application, setApplication] = useState('generic');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const navigate = useNavigate();

  const applications = [
    'generic', 'healthcare', 'finance', 'insurance', 'gov', 'legal',
    'ads', 'gaming', 'realtime', 'edge', 'mobile', 'iot',
    'marketing', 'retail', 'logistics', 'manufacturing', 'energy', 'telco',
    'cybersecurity', 'agriculture', 'education', 'sports',
    'retail & e-commerce', 'manufacturing & iot', 'smart cities & transport',
    'energy & environment', 'social media & nlp'
  ];

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const files = e.dataTransfer.files;
    if (files && files[0]) setSelectedFile(files[0]);
  }, []);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file first.');
      return;
    }

    if (!application) {
      alert('Please select an application domain.');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      const response = await api.upload(selectedFile, application, setUploadProgress);
      console.log("✅ Upload Response:", response);
      if (onUploadComplete) onUploadComplete(response); // Fix: no response.data
      navigate('/training');
    } catch (error) {
      console.error("❌ Upload failed:", error.response?.data || error.message);
      alert('Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <section className="upload-section">
      <div className="container">
        <div className="upload-container">
          <h1>Upload Your Dataset</h1>
          <p>Start by uploading your dataset — CSV, Excel, JSON, Parquet, or ZIP.</p>

          <div
            className={`upload-area ${dragActive ? 'dragover' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            <input
              id="file-input"
              type="file"
              style={{ display: 'none' }}
              onChange={handleFileSelect}
              accept=".csv,.json,.xlsx,.xls,.parquet,.zip"
            />
            <p>Drag & drop your file or click to browse</p>
            <div className="file-formats">
              <span className="format-tag">CSV</span>
              <span className="format-tag">Excel</span>
              <span className="format-tag">JSON</span>
              <span className="format-tag">ZIP</span>
            </div>
            {selectedFile && (
              <p style={{ marginTop: '1rem', color: '#007bff' }}>
                Selected: {selectedFile.name}
              </p>
            )}
          </div>

          {uploading && (
            <div style={{ marginTop: '1rem', width: '100%' }}>
              <p>Uploading: {uploadProgress.toFixed(0)}%</p>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </div>
          )}

          <select
            className="application-select"
            value={application}
            onChange={(e) => setApplication(e.target.value)}
          >
            <option value="">Select Application Domain</option>
            {applications.map(app => (
              <option key={app} value={app}>
                {app.charAt(0).toUpperCase() + app.slice(1)}
              </option>
            ))}
          </select>

          <button
            className="btn btn-primary"
            onClick={handleUpload}
            disabled={uploading || !selectedFile}
          >
            {uploading ? 'Uploading...' : 'Start Training'}
          </button>
        </div>
      </div>
    </section>
  );
};

export default Upload;