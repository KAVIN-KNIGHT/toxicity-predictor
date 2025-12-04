import React, { useState } from 'react';
import './PredictionForm.css';

const SAMPLE_MOLECULES = [
  { name: 'Ethanol', smiles: 'CCO' },
  { name: 'Benzene', smiles: 'c1ccccc1' },
  { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
  { name: 'Bisphenol A', smiles: 'CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O' },
  { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
  { name: 'TCDD (Dioxin)', smiles: 'O1c2c(Cl)c(Cl)cc3c2Oc4c1c(Cl)c(Cl)cc4O3' },
  { name: 'Aflatoxin B1', smiles: 'COc1cc2c(c3c1oc(=O)c1c3ccc3c1C(=O)OC3)C1C=COC1O2' },
  { name: 'Paraquat', smiles: 'C[n+]1ccc(cc1)c2cc[n+](C)cc2' },
  { name: 'Strychnine', smiles: 'O=C1N2C3C(C(=O)CC2)C4N(C5C3C6=CC=CC=C6N(C45)C)C=C1' },
  { name: 'DDT', smiles: 'ClC(C(C1=CC=C(Cl)C=C1)C2=CC=C(Cl)C=C2)Cl' }
];

function PredictionForm({ onPredict, loading }) {
  const [smiles, setSmiles] = useState('');
  const [compoundName, setCompoundName] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (smiles.trim()) {
      onPredict(smiles.trim(), compoundName.trim() || 'Unknown Compound');
    }
  };

  const loadSample = (sample) => {
    setSmiles(sample.smiles);
    setCompoundName(sample.name);
  };

  return (
    <div className="prediction-form-container">
      <h2>Enter Compound Information</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="smiles">SMILES String *</label>
          <input
            type="text"
            id="smiles"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            placeholder="e.g., CCO or c1ccccc1"
            required
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="compoundName">Compound Name (Optional)</label>
          <input
            type="text"
            id="compoundName"
            value={compoundName}
            onChange={(e) => setCompoundName(e.target.value)}
            placeholder="e.g., Ethanol"
            disabled={loading}
          />
        </div>

        <button type="submit" disabled={loading || !smiles.trim()}>
          {loading ? 'Analyzing...' : 'Predict Toxicity'}
        </button>
      </form>

      <div className="quick-select">
        <h3>Quick Select</h3>
        <div className="sample-buttons">
          {SAMPLE_MOLECULES.map((sample) => (
            <button
              key={sample.name}
              type="button"
              onClick={() => loadSample(sample)}
              disabled={loading}
              className="sample-btn"
            >
              {sample.name}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default PredictionForm;
