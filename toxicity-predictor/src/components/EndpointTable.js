import React from 'react';
import './EndpointTable.css';

function EndpointTable({ predictions }) {
  const getRiskEmoji = (riskLevel) => {
    switch (riskLevel) {
      case 'LOW':
        return '🟢';
      case 'MODERATE':
        return '🟡';
      case 'HIGH':
        return '🔴';
      default:
        return '⚪';
    }
  };

  return (
    <div className="endpoint-table-container">
      <h2>Toxicity Endpoint Predictions</h2>
      <div className="table-wrapper">
        <table className="endpoint-table">
          <thead>
            <tr>
              <th>Endpoint</th>
              <th>Prediction</th>
              <th>Probability</th>
              <th>Risk Level</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((pred, index) => (
              <tr key={index} className={pred.prediction === 'TOXIC' ? 'toxic-row' : ''}>
                <td className="endpoint-name">{pred.endpoint}</td>
                <td className={`prediction ${pred.prediction.toLowerCase()}`}>
                  {pred.prediction}
                </td>
                <td className="probability">{pred.probability}%</td>
                <td className="risk-level">
                  {getRiskEmoji(pred.risk_level)} {pred.risk_level}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default EndpointTable;
