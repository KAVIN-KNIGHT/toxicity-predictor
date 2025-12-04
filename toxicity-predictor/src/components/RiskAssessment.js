import React from 'react';
import './RiskAssessment.css';

function RiskAssessment({ assessment }) {
  // Use the risk level and color from backend (based on max probability logic)
  const riskColor = assessment?.risk_color || '#16a34a';
  const riskLevel = assessment?.risk_level || 'LOW';
  const riskExplanation = assessment?.risk_explanation || 'Assessment complete';
  const maxProbability = assessment?.max_probability ?? 0;
  const avgProbability = assessment?.average_probability ?? 0;
  const highRiskEndpoints = assessment?.high_risk_endpoints ?? 0;
  const toxicEndpoints = assessment?.toxic_endpoints ?? 0;
  const totalEndpoints = assessment?.total_endpoints ?? 12;

  return (
    <div className="risk-assessment">
      <h2>Overall Risk Assessment</h2>
      <div className="risk-content">
        <div 
          className="risk-badge"
          style={{ backgroundColor: riskColor }}
        >
          {riskLevel} RISK
        </div>
        <div className="risk-details">
          <div className="risk-explanation">
            <span className="explanation-text">{riskExplanation}</span>
          </div>
          <div className="risk-stat">
            <span className="stat-label">Maximum Toxicity Probability</span>
            <span className="stat-value" style={{ color: riskColor, fontWeight: 'bold' }}>
              {maxProbability}%
            </span>
          </div>
          <div className="risk-stat">
            <span className="stat-label">Average Toxicity Probability</span>
            <span className="stat-value">{avgProbability}%</span>
          </div>
          <div className="risk-stat">
            <span className="stat-label">High Risk Endpoints</span>
            <span className="stat-value" style={{ color: highRiskEndpoints >= 2 ? '#dc2626' : '#16a34a' }}>
              {highRiskEndpoints} / {totalEndpoints}
            </span>
          </div>
          <div className="risk-stat">
            <span className="stat-label">Total Toxic Endpoints</span>
            <span className="stat-value">{toxicEndpoints} / {totalEndpoints}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default RiskAssessment;
