import React from 'react';
import './Visualization.css';

function Visualization({ visualization, compoundName }) {
  return (
    <div className="visualization-container">
      <h2>Molecular Analysis - {compoundName}</h2>
      <div className="visualization-image">
        <img 
          src={`data:image/png;base64,${visualization}`} 
          alt={`${compoundName} molecular structure and analysis`}
        />
      </div>
      <div className="visualization-legend">
        <div className="legend-item">
          <strong>Left Panel:</strong> Molecular Structure (2D representation)
        </div>
        <div className="legend-item">
          <strong>Center Panel:</strong> Morgan Fingerprint Heatmap (2048-bit features)
        </div>
        <div className="legend-item">
          <strong>Right Panel:</strong> Molecular Graph (atoms as nodes, bonds as edges)
        </div>
      </div>
    </div>
  );
}

export default Visualization;
