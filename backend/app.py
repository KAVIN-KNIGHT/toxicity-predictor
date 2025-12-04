"""
Flask API for Hybrid Toxicity Prediction Model
Provides endpoints for React frontend to get toxicity predictions with visualizations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool
from torch import nn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx
import io
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Toxicity endpoints
ENDPOINTS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# Global variables for models
model_dict = None
gnn_model = None

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def smiles_to_graph(smiles):
    """Convert SMILES to PyTorch Geometric graph"""
    from rdkit.Chem import rdchem
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Define vocabularies
    ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    HYBRIDIZATION_LIST = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3]
    CHIRALITY_LIST = [rdchem.CHI_UNSPECIFIED, rdchem.CHI_TETRAHEDRAL_CW, rdchem.CHI_TETRAHEDRAL_CCW]
    
    def one_hot_encoding(value, choices):
        vec = [0] * len(choices)
        if value in choices:
            vec[choices.index(value)] = 1
        return vec
    
    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = (
            one_hot_encoding(atom.GetAtomicNum(), ATOM_LIST) +
            one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION_LIST) +
            [atom.GetTotalNumHs(includeNeighbors=True), atom.GetFormalCharge(), int(atom.GetIsAromatic())] +
            one_hot_encoding(atom.GetChiralTag(), CHIRALITY_LIST)
        )
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Bond features
    BOND_TYPES = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]
    
    def bond_features(bond):
        bt = bond.GetBondType()
        return one_hot_encoding(bt, BOND_TYPES) + [int(bond.GetIsConjugated()), int(bond.IsInRing())]
    
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)
    
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_TYPES) + 2), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# GNN Model Architecture
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class MultiLabelGIN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=4, num_tasks=12, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GINConv(MLP(in_dim, hidden_dim, hidden_dim)))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(MLP(hidden_dim, hidden_dim, hidden_dim)))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
        x = global_add_pool(x, batch)
        x = self.dropout(x)
        logits = self.head(x)
        return logits

def generate_visualization(smiles, fp, graph):
    """Generate base64 encoded visualization image"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mol_weight = Descriptors.MolWt(mol)
    
    fig = plt.figure(figsize=(16, 5))
    
    # 1. Molecular Structure
    ax1 = plt.subplot(1, 3, 1)
    img = Draw.MolToImage(mol, size=(400, 400))
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'Molecular Structure\n{formula} (MW: {mol_weight:.2f})', fontsize=12, fontweight='bold')
    
    # 2. Morgan Fingerprint Heatmap
    ax2 = plt.subplot(1, 3, 2)
    fp_grid = fp.reshape(32, 64)
    im = ax2.imshow(fp_grid, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax2.set_title('Morgan Fingerprint (2048 bits)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Bit Index (cols)')
    ax2.set_ylabel('Block (rows)')
    plt.colorbar(im, ax=ax2, label='Bit Value')
    
    # 3. Molecular Graph
    ax3 = plt.subplot(1, 3, 3)
    G = to_networkx(graph, to_undirected=True)
    
    from rdkit.Chem import rdchem
    ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    node_atomic_nums = []
    for i in range(graph.x.size(0)):
        atom_slice = graph.x[i, :len(ATOM_LIST)].numpy()
        if atom_slice.sum() == 1:
            pos = int(np.argmax(atom_slice))
            atomic_num = ATOM_LIST[pos]
        else:
            atomic_num = 0
        node_atomic_nums.append(atomic_num)
    
    pt = Chem.GetPeriodicTable()
    node_labels = {i: (pt.GetElementSymbol(Z) if Z > 0 else '?') for i, Z in enumerate(node_atomic_nums)}
    
    unique_Z = sorted(set(node_atomic_nums))
    color_map_dict = {Z: idx for idx, Z in enumerate(unique_Z)}
    node_colors = [color_map_dict[Z] for Z in node_atomic_nums]
    
    pos = nx.spring_layout(G, seed=42, k=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='Set3', 
                           node_size=500, ax=ax3, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, ax=ax3, alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, 
                            font_weight='bold', ax=ax3)
    
    ax3.set_title('Molecular Graph (GNN Input)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64

def load_models():
    """Load all models on startup"""
    global model_dict, gnn_model
    
    try:
        model_dict = joblib.load(r"C:\Users\admin\Desktop\Toxicity prediction\backend\FinalModel.pkl")
        
        # Initialize GNN model
        sample_graph = smiles_to_graph("CCO")
        in_dim = sample_graph.x.shape[1]
        gnn_model = MultiLabelGIN(in_dim=in_dim, hidden_dim=160, num_layers=5, num_tasks=len(ENDPOINTS))
        gnn_model.load_state_dict(model_dict['gnn_state_dict'])
        gnn_model.eval()
        
        print("✅ Models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_dict is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expects JSON: { "smiles": "CCO", "compound_name": "Ethanol" }
    Returns: predictions, probabilities, molecular properties, and visualization
    """
    try:
        data = request.get_json()
        smiles = data.get('smiles', '').strip()
        compound_name = data.get('compound_name', 'Unknown Compound')
        
        if not smiles:
            return jsonify({'error': 'SMILES string is required'}), 400
        
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({'error': 'Invalid SMILES string'}), 400
        
        # Get molecular properties
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mol_weight = Descriptors.MolWt(mol)
        mol_logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        
        # Generate fingerprint and graph
        fp = smiles_to_fingerprint(smiles)
        graph = smiles_to_graph(smiles)
        
        if fp is None or graph is None:
            return jsonify({'error': 'Could not generate molecular features'}), 400
        
        # Get base model predictions
        rf_models = [model_dict['rf_models'][ep] for ep in ENDPOINTS]
        adaboost_models = [model_dict['adaboost_models'][ep] for ep in ENDPOINTS]
        
        rf_probs = []
        for clf in rf_models:
            prob = clf.predict_proba(fp.reshape(1, -1))[0, 1]
            rf_probs.append(prob)
        rf_probs = np.array(rf_probs)
        
        ada_probs = []
        for clf in adaboost_models:
            prob = clf.predict_proba(fp.reshape(1, -1))[0, 1]
            ada_probs.append(prob)
        ada_probs = np.array(ada_probs)
        
        # GNN predictions
        with torch.no_grad():
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
            gnn_output = gnn_model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            gnn_probs = torch.sigmoid(gnn_output).numpy()[0]
        
        # Meta-model predictions
        stacked_probs = np.column_stack([rf_probs, ada_probs, gnn_probs])
        meta_models = [model_dict['meta_models'][ep] for ep in ENDPOINTS if ep in model_dict['meta_models']]
        
        final_predictions = []
        final_probabilities = []
        for i, meta_clf in enumerate(meta_models):
            prob = meta_clf.predict_proba(stacked_probs[i:i+1])[0, 1]
            pred = 1 if prob >= 0.5 else 0
            final_predictions.append(pred)
            final_probabilities.append(float(prob))
        
        # Calculate overall metrics
        avg_prob = float(np.mean(final_probabilities))
        max_prob = float(np.max(final_probabilities))
        toxic_count = sum(final_predictions)
        high_risk_count = sum(1 for p in final_probabilities if p >= 0.7)

        # Risk assessment based on max probability
        if max_prob >= 0.7:
            overall_risk = "HIGH"
            risk_color = "#dc2626"
            risk_explanation = f"Compound shows high toxicity risk with maximum probability of {max_prob*100:.1f}%"
        elif max_prob >= 0.3:
            overall_risk = "MODERATE"
            risk_color = "#f59e0b"
            risk_explanation = f"Compound shows moderate toxicity risk with maximum probability of {max_prob*100:.1f}%"
        else:
            overall_risk = "LOW"
            risk_color = "#16a34a"
            risk_explanation = f"Compound shows low toxicity risk with maximum probability of {max_prob*100:.1f}%"

        # Generate visualization
        visualization_base64 = generate_visualization(smiles, fp, graph)
        
        # Prepare endpoint predictions table
        endpoint_predictions = []
        for i, endpoint in enumerate(ENDPOINTS[:len(final_probabilities)]):
            prob = final_probabilities[i]
            pred = final_predictions[i]
            
            if prob >= 0.7:
                risk_level = "HIGH"
                risk_emoji = "🔴"
            elif prob >= 0.3:
                risk_level = "MEDIUM"
                risk_emoji = "🟡"
            else:
                risk_level = "LOW"
                risk_emoji = "🟢"
            
            endpoint_predictions.append({
                'endpoint': endpoint,
                'prediction': 'TOXIC' if pred == 1 else 'NON-TOXIC',
                'probability': round(prob * 100, 2),
                'risk_level': risk_level,
                'risk_emoji': risk_emoji
            })
        
        # Build response
        response = {
            'success': True,
            'compound_name': compound_name,
            'smiles': smiles,
            'molecular_properties': {
                'formula': formula,
                'molecular_weight': round(mol_weight, 2),
                'logP': round(mol_logp, 2),
                'tpsa': round(tpsa, 2),
                'h_donors': h_donors,
                'h_acceptors': h_acceptors
            },
            'overall_assessment': {
                'average_probability': round(avg_prob * 100, 2),
                'max_probability': round(max_prob * 100, 2),
                'toxic_endpoints': toxic_count,
                'total_endpoints': len(ENDPOINTS),
                'high_risk_endpoints': high_risk_count,
                'risk_level': overall_risk,
                'risk_color': risk_color,
                'risk_explanation': risk_explanation
            },
            'endpoint_predictions': endpoint_predictions,
            'visualization': visualization_base64,
            'base_model_contributions': {
                'random_forest': [round(p * 100, 2) for p in rf_probs.tolist()],
                'adaboost': [round(p * 100, 2) for p in ada_probs.tolist()],
                'gnn': [round(p * 100, 2) for p in gnn_probs.tolist()]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    Expects JSON: { "compounds": [{"smiles": "CCO", "name": "Ethanol"}, ...] }
    """
    try:
        data = request.get_json()
        compounds = data.get('compounds', [])
        
        if not compounds or len(compounds) == 0:
            return jsonify({'error': 'No compounds provided'}), 400
        
        results = []
        for compound in compounds[:10]:  # Limit to 10 compounds
            smiles = compound.get('smiles', '').strip()
            name = compound.get('name', 'Unknown')
            
            if smiles:
                # Use the predict endpoint logic
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Simplified batch processing without visualization
                    fp = smiles_to_fingerprint(smiles)
                    graph = smiles_to_graph(smiles)
                    
                    if fp is not None and graph is not None:
                        # Quick predictions
                        rf_models = [model_dict['rf_models'][ep] for ep in ENDPOINTS]
                        rf_probs = np.array([clf.predict_proba(fp.reshape(1, -1))[0, 1] for clf in rf_models])
                        
                        ada_models = [model_dict['adaboost_models'][ep] for ep in ENDPOINTS]
                        ada_probs = np.array([clf.predict_proba(fp.reshape(1, -1))[0, 1] for clf in ada_models])
                        
                        with torch.no_grad():
                            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
                            gnn_output = gnn_model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                            gnn_probs = torch.sigmoid(gnn_output).numpy()[0]
                        
                        stacked = np.column_stack([rf_probs, ada_probs, gnn_probs])
                        meta_models = [model_dict['meta_models'][ep] for ep in ENDPOINTS if ep in model_dict['meta_models']]
                        
                        probs = [meta_clf.predict_proba(stacked[i:i+1])[0, 1] for i, meta_clf in enumerate(meta_models)]
                        avg_prob = float(np.mean(probs))
                        toxic_count = sum([1 if p >= 0.5 else 0 for p in probs])
                        
                        results.append({
                            'name': name,
                            'smiles': smiles,
                            'average_toxicity': round(avg_prob * 100, 2),
                            'toxic_endpoints': toxic_count,
                            'total_endpoints': len(ENDPOINTS)
                        })
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading models...")
    if load_models():
        print("\n🚀 Starting Flask API server...")
        print("API will be available at http://localhost:5000")
        print("\nEndpoints:")
        print("  GET  /api/health        - Health check")
        print("  POST /api/predict       - Single compound prediction")
        print("  POST /api/batch-predict - Batch prediction")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models. Please ensure FinalModel.pkl exists.")
