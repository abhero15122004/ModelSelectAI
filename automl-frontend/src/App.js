import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Hero from './components/Hero';
import Upload from './components/Upload';
import Training from './components/Training';
import Results from './components/Results';
import './App.css';

function App() {
  const [currentRun, setCurrentRun] = useState(null);
  const [uploadData, setUploadData] = useState(null);

  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Hero />} />
          <Route 
            path="/upload" 
            element={
              <Upload 
                onUploadComplete={(data) => {
                  setUploadData(data);
                  setCurrentRun(data.run_id);
                }} 
              />
            } 
          />
          <Route 
            path="/training" 
            element={
              <Training 
                runId={currentRun}
                uploadData={uploadData}
              />
            } 
          />
          <Route 
            path="/results" 
            element={
              <Results 
                runId={currentRun}
                uploadData={uploadData}
              />
            } 
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;