import React from 'react';
import './MolecularProperties.css';

function MolecularProperties({ properties, compoundName }) {
  return (
    <div className="molecular-properties">
      <h2>Molecular Properties - {compoundName}</h2>
      <div className="properties-grid">
        <div className="property-item">
          <span className="property-label">Formula</span>
          <span className="property-value">{properties.formula}</span>
        </div>
        <div className="property-item">
          <span className="property-label">Molecular Weight</span>
          <span className="property-value">{properties.molecular_weight} g/mol</span>
        </div>
        <div className="property-item">
          <span className="property-label">LogP</span>
          <span className="property-value">{properties.logP}</span>
        </div>
        <div className="property-item">
          <span className="property-label">TPSA</span>
          <span className="property-value">{properties.tpsa} Ų</span>
        </div>
        <div className="property-item">
          <span className="property-label">H-Bond Donors</span>
          <span className="property-value">{properties.h_donors}</span>
        </div>
        <div className="property-item">
          <span className="property-label">H-Bond Acceptors</span>
          <span className="property-value">{properties.h_acceptors}</span>
        </div>
      </div>
    </div>
  );
}

export default MolecularProperties;
