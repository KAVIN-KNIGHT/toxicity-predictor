# 🧪 Toxicity Prediction System

A comprehensive machine learning system for predicting chemical toxicity across 12 different biological endpoints using a hybrid ensemble of Random Forest, AdaBoost, and Graph Neural Networks.

## 📋 Project Overview

This project combines three complementary ML models to predict toxicity of chemical compounds based on SMILES strings:

- **Random Forest (RF)**: Fast, interpretable, ~91.87% accuracy
- **AdaBoost**: Boosting ensemble approach
- **Graph Neural Network (GNN)**: Deep learning on molecular graphs

The system processes molecular structures, generates fingerprints, and provides comprehensive toxicity assessments across 12 toxicity endpoints from the [Tox21 dataset](https://tox21.gov/).

## 🏗️ Project Structure

```
Toxicity-predictor/
├── backend/                          # Flask API & ML models
│   ├── app.py                        # Flask API server
│   ├── project.ipynb                 # Model training & development
│   ├── results.ipynb                 # Results analysis
│   ├── requirements.txt              # Python dependencies
│   ├── tox21.csv                     # Dataset
│   ├── X_train.npy, X_test.npy       # Fingerprint data
│   ├── y_train.npy, y_test.npy       # Labels
│   └── *.pkl                         # Trained models (RF, AdaBoost, GNN)
│
├── toxicity-predictor/               # React frontend
│   ├── src/
│   │   ├── App.js                    # Main application
│   │   ├── App.css                   # Application styling
│   │   ├── components/
│   │   │   ├── PredictionForm.js     # SMILES input form
│   │   │   ├── MolecularProperties.js # Property display
│   │   │   ├── RiskAssessment.js     # Risk level display
│   │   │   ├── EndpointTable.js      # Toxicity predictions table
│   │   │   ├── Visualization.js      # Molecular visualization
│   │   │   └── *.css                 # Component styles
│   │   └── index.js                  # React entry point
│   ├── public/
│   │   └── index.html
│   ├── package.json                  # Node dependencies
│   └── build/                        # Production build
│
├── tox21.csv                         # Original dataset
└── .gitignore                        # Git ignore rules
```

## 🎯 Toxicity Endpoints

The system predicts toxicity for 12 endpoints:

| Endpoint | Type | Description |
|----------|------|-------------|
| NR-AR | Nuclear Receptor | Androgen Receptor |
| NR-AR-LBD | Nuclear Receptor | AR Ligand Binding Domain |
| NR-AhR | Nuclear Receptor | Aryl Hydrocarbon Receptor |
| NR-Aromatase | Nuclear Receptor | Aromatase |
| NR-ER | Nuclear Receptor | Estrogen Receptor |
| NR-ER-LBD | Nuclear Receptor | ER Ligand Binding Domain |
| NR-PPAR-gamma | Nuclear Receptor | PPAR-Gamma |
| SR-ARE | Stress Response | Antioxidant Response Element |
| SR-ATAD5 | Stress Response | DNA Damage Inducible 45 Alpha |
| SR-HSE | Stress Response | Heat Shock Element |
| SR-MMP | Stress Response | Mitochondrial Membrane Potential |
| SR-p53 | Stress Response | p53 Response |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. **Install Python Dependencies**
```bash
cd backend
pip install -r requirements.txt
```

2. **Ensure Model Files Exist**
The following files should be in the `backend` directory:
- `FinalModel.pkl` (combined trained models)
- `X_train.npy`, `X_test.npy` (fingerprint data)
- `y_train.npy`, `y_test.npy` (labels)

If models don't exist, run the training notebook:
```bash
jupyter notebook project.ipynb
```

3. **Start Flask API Server**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Frontend Setup

1. **Install Node Dependencies**
```bash
cd toxicity-predictor
npm install
```

2. **Start React Development Server**
```bash
npm start
```

The application will open at `http://localhost:3000`

## 📊 API Endpoints

### Health Check
```
GET /api/health
```

### Single Prediction
```
POST /api/predict
Content-Type: application/json

{
  "smiles": "CCO",
  "compound_name": "Ethanol"
}
```

**Response:**
```json
{
  "success": true,
  "compound_name": "Ethanol",
  "smiles": "CCO",
  "molecular_properties": {
    "formula": "C2H6O",
    "molecular_weight": 46.07,
    "logP": -0.31,
    "tpsa": 20.23,
    "h_donors": 1,
    "h_acceptors": 1
  },
  "overall_assessment": {
    "average_probability": 15.34,
    "max_probability": 42.15,
    "toxic_endpoints": 2,
    "total_endpoints": 12,
    "high_risk_endpoints": 0,
    "risk_level": "LOW",
    "risk_color": "#16a34a",
    "risk_explanation": "Low toxicity risk"
  },
  "endpoint_predictions": [...],
  "visualization": "base64_encoded_image",
  "base_model_contributions": {...}
}
```

### Batch Prediction
```
POST /api/batch-predict
Content-Type: application/json

{
  "smiles_list": ["CCO", "c1ccccc1", "CC(=O)O"],
  "compound_names": ["Ethanol", "Benzene", "Acetic Acid"]
}
```

## 🧬 Key Features

### 1. **Molecular Visualization**
- 2D molecular structure rendering using RDKit
- Morgan fingerprint heatmaps (2048-bit)
- Molecular graph visualization with atomic labels

### 2. **Comprehensive Molecular Properties**
- Molecular Formula
- Molecular Weight (Da)
- LogP (lipophilicity)
- TPSA (polar surface area)
- H-Bond Donors/Acceptors
- Ring count

### 3. **Hybrid Predictions**
- **Random Forest**: Primary model, most reliable
- **AdaBoost**: Comparative analysis
- **GNN**: Deep learning perspective
- Meta-learner ensemble combines all three

### 4. **Risk Assessment**
- **HIGH RISK** (red): max probability ≥ 70%
- **MEDIUM RISK** (yellow): max probability ≥ 30%
- **LOW RISK** (green): max probability < 30%

### 5. **Interactive UI**
- Quick sample molecules for testing
- Real-time SMILES validation
- Responsive design for all screen sizes
- Detailed endpoint-by-endpoint results

## 📈 Model Performance

### Random Forest (Recommended)
- **Average Accuracy**: 91.87%
- **Average F1-Score**: 43.77%
- **Average ROC-AUC**: 0.807

### AdaBoost
- **Average Accuracy**: 78.5%
- **Average F1-Score**: 25.3%
- **Note**: Lower performance on imbalanced endpoints

### Graph Neural Network
- **Average Accuracy**: 93.2%
- **Variable performance**: Excellent on some endpoints, poor on others
- **Best for**: Complex molecular structures

## 💾 Data Format

### Input
```python
SMILES string format:
- "CCO" (Ethanol)
- "c1ccccc1" (Benzene)
- "CC(=O)O" (Acetic Acid)
- "O=C1c2ccccc2C(=O)C1c1ccc2cc(S(=O)(=O)[O-])cc(S(=O)(=O)[O-])c2n1" (Complex molecule)
```

### Output Features
- 2048-bit Morgan fingerprints (radius=2)
- Per-endpoint binary/probabilistic predictions
- Confidence scores (0-100%)
- Risk classifications

## 🔧 Configuration

### Environment Variables (Optional)
Create a `.env` file in the backend directory:
```
FLASK_ENV=development
FLASK_DEBUG=true
PORT=5000
```

### Model Parameters
All configurable in `backend/app.py`:
```python
ENDPOINTS = ['NR-AR', 'NR-AR-LBD', ...]  # Toxicity endpoints
FINGERPRINT_SIZE = 2048                   # Morgan FP bits
FINGERPRINT_RADIUS = 2                    # Morgan FP radius
```

## 📚 Training & Development

### Retraining Models
Open `backend/project.ipynb`:
```bash
cd backend
jupyter notebook project.ipynb
```

**Notebook Sections:**
1. Data loading & exploration
2. Fingerprint generation
3. Train/test split
4. Random Forest training
5. AdaBoost training
6. ANN training
7. GNN training
8. Model evaluation
9. Meta-learner creation

### Results Analysis
See `backend/results.ipynb` for:
- Detailed accuracy comparisons
- Per-endpoint performance metrics
- Confusion matrices
- ROC-AUC curves
- Model strengths/weaknesses

