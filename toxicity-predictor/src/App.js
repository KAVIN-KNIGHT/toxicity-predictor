import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import PredictionForm from './components/PredictionForm';
import MolecularProperties from './components/MolecularProperties';
import RiskAssessment from './components/RiskAssessment';
import EndpointTable from './components/EndpointTable';
import Visualization from './components/Visualization';

const API_URL = 'http://localhost:5000/api';

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handlePredict = async (smiles, compoundName) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        smiles: smiles,
        compound_name: compoundName
      });

      if (response.data.success) {
        setResult(response.data);
      } else {
        setError(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      if (err.response) {
        setError(err.response.data.error || 'Server error');
      } else if (err.request) {
        setError('Cannot connect to API server. Make sure Flask server is running at http://localhost:5000');
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧪 Toxicity Prediction System</h1>
        <p>Hybrid AI Model: Random Forest + AdaBoost + Graph Neural Network</p>
      </header>

      <main className="App-main">
        <PredictionForm onPredict={handlePredict} loading={loading} />

        {error && (
          <div className="error-card">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {loading && (
          <div className="loading-card">
            <div className="spinner"></div>
            <p>Analyzing compound... This may take a few seconds.</p>
          </div>
        )}

        {result && (
          <div className="results-container">
            <MolecularProperties properties={result.molecular_properties} compoundName={result.compound_name} />
            <RiskAssessment assessment={result.overall_assessment} />
            <EndpointTable predictions={result.endpoint_predictions} />
            <Visualization visualization={result.visualization} compoundName={result.compound_name} />
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Powered by Hybrid ML Ensemble • 12 Toxicity Endpoints • ML Project 2025</p>
      </footer>
    </div>
  );
}

export default App;
