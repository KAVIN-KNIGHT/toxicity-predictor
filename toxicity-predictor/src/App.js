import React, { useState } from 'react';
import './App.css';
import PredictionForm from './components/PredictionForm';
import MolecularProperties from './components/MolecularProperties';
import RiskAssessment from './components/RiskAssessment';
import EndpointTable from './components/EndpointTable';
import Visualization from './components/Visualization';
import { predictToxicity, formatApiError } from './api';

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handlePredict = async (smiles, compoundName) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictToxicity({
        smiles,
        compoundName
      });

      if (data.success) {
        setResult(data);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (err) {
      setError(formatApiError(err));
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
