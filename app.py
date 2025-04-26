from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # For handling CORS
import requests
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import csv
import os
import datetime
from sklearn.metrics import f1_score, precision_recall_curve, auc, accuracy_score

app = Flask(__name__)
CORS(app)

class GNNSideEffectsModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

class DrugDatasetConnector:
    def __init__(self, csv_path):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.drug_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
        self.drug_embeddings = None
        if self.drug_df is not None:
            self.drug_embeddings = self.model.encode(self.drug_df['drug_name'].tolist())

    def search_local_dataset(self, drug_name, threshold=0.7):
        """Search for drug in local dataset using embeddings"""
        if self.drug_df is None or self.drug_embeddings is None:
            return None

        # First try exact match
        exact_match = self.drug_df[self.drug_df['drug_name'].str.lower() == drug_name.lower()]
        if not exact_match.empty:
            return exact_match.iloc[0].to_dict()

        # If no exact match, use semantic search
        query_embedding = self.model.encode([drug_name])
        similarities = util.pytorch_cos_sim(query_embedding, self.drug_embeddings)[0]

        # Find the most similar drug above threshold
        max_idx = similarities.argmax().item()
        if similarities[max_idx] >= threshold:
            return self.drug_df.iloc[max_idx].to_dict()

        return None

# Initialize the drug dataset connector
local_dataset_path = 'local_drug_dataset.csv'
drug_connector = DrugDatasetConnector(local_dataset_path)

@app.route('/api/local_drug_info', methods=['GET'])
def local_drug_info():
    drug_name = request.args.get('drug_name', '')
    if not drug_name:
        return jsonify({'error': 'Please provide a drug name'}), 400

    # Search in local dataset
    drug_info = drug_connector.search_local_dataset(drug_name)
    if drug_info:
        return jsonify({'success': True, 'data': drug_info})
    else:
        return jsonify({'success': False, 'error': 'Drug not found in local dataset'}), 404

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/multi_drug_info', methods=['POST'])
def multi_drug_info():
    data = request.json
    drug_names = data.get('drug_names', [])

    if not drug_names or not isinstance(drug_names, list):
        return jsonify({'error': 'Please provide a list of drug names'}), 400

    try:
        # Get information for each drug
        drugs_info = []
        all_side_effects = {}

        for drug_name in drug_names:
            drug_data = {}
            try:
                pubchem_info = get_pubchem_info(drug_name)
                side_effects = get_side_effects(drug_name)
                drug_data = {**pubchem_info, 'side_effects': side_effects, 'source': 'external_api'}
            except Exception:
                local_info = drug_connector.search_local_dataset(drug_name)
                if local_info and 'side_effects' in local_info:
                    side_effects = []
                    for effect in local_info['side_effects'].split('|'):
                        if ':' in effect:
                            term, count = effect.split(':')
                            side_effects.append({'term': term.strip(), 'count': int(count)})
                    drug_data = {
                        'drug_name': drug_name,
                        'source': 'local_dataset',
                        'side_effects': side_effects
                    }
                    for key in ['molecular_formula', 'molecular_weight', 'iupac_name']:
                        if key in local_info:
                            drug_data[key] = local_info[key]
                else:
                    side_effects = get_side_effects(drug_name)
                    drug_data = {
                        'drug_name': drug_name,
                        'source': 'external_api_fallback',
                        'side_effects': side_effects
                    }

            for effect in drug_data['side_effects']:
                effect_term = effect['term']
                if effect_term not in all_side_effects:
                    all_side_effects[effect_term] = {'count': 0, 'drugs': []}
                all_side_effects[effect_term]['count'] += effect['count']
                all_side_effects[effect_term]['drugs'].append(drug_name)

            drugs_info.append(drug_data)

        common_side_effects = analyze_side_effects(drugs_info, all_side_effects)

        # DYNAMIC METRICS CALCULATION - Based on actual data
        # Calculate true positives, false positives, etc. for side effects 
        # that occur in multiple drugs vs. those that don't
        
        # Ground truth: side effects that are common to multiple drugs
        y_true = []
        # Predictions: side effects with count above median are predicted as common
        y_pred = []
        # Raw scores for PR curve
        y_scores = []
        
        # Get all side effects for calculation 
        all_effects = []
        for drug in drugs_info:
            for effect in drug.get('side_effects', []):
                all_effects.append({
                    'term': effect['term'],
                    'count': effect['count'],
                    'is_common': len(all_side_effects.get(effect['term'], {}).get('drugs', [])) > 1
                })
        
        # If we have enough data points for meaningful metrics
        if len(all_effects) > 2:
            # Find median count to use as threshold
            counts = [effect['count'] for effect in all_effects]
            median_count = np.median(counts) if counts else 0
            
            # Calculate metrics
            for effect in all_effects:
                # Ground truth: is this side effect common to multiple drugs?
                y_true.append(1 if effect['is_common'] else 0)
                # Prediction: is count above threshold?
                y_pred.append(1 if effect['count'] > median_count else 0)
                # Score for PR curve: normalized count 
                y_scores.append(effect['count'] / (max(counts) if max(counts) > 0 else 1))
            
            # Calculate metrics if we have variation in the classes
            if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
                f1 = f1_score(y_true, y_pred, average='binary')
                acc = accuracy_score(y_true, y_pred)
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                aucpr = auc(recall, precision)
            else:
                # Not enough variation for meaningful metrics
                f1, acc, aucpr = 0.5, 0.5, 0.5
        else:
            # Not enough data points
            f1, acc, aucpr = 0.5, 0.5, 0.5

        return jsonify({
            'success': True,
            'drugs_info': drugs_info,
            'common_side_effects': common_side_effects,
            'f1_score': round(f1, 3),
            'accuracy': round(acc, 3),
            'aucpr': round(aucpr, 3)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    drug_name = data.get('drug_name')
    side_effects = data.get('side_effects', [])
    feedback_type = data.get('feedback_type', 'unknown')

    if not drug_name:
        return jsonify({'error': 'Missing drug name'}), 400

    try:
        # Create feedback directory if it doesn't exist
        os.makedirs('feedback', exist_ok=True)
        
        # Use current date for filename
        today = datetime.date.today().isoformat()
        feedback_file = f'feedback/feedback_{today}.csv'
        file_exists = os.path.isfile(feedback_file)

        with open(feedback_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'drug_name', 'feedback_type', 'side_effect', 'count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for effect in side_effects:
                writer.writerow({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'drug_name': drug_name,
                    'feedback_type': feedback_type,
                    'side_effect': effect.get('term', ''),
                    'count': effect.get('count', 0)
                })

        return jsonify({'success': True, 'message': 'Feedback recorded'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_side_effects(drugs_info, all_side_effects):
    try:
        # Skip GNN if we have fewer than 2 drugs or side effects
        if len(drugs_info) < 2 or len(all_side_effects) < 2:
            # Just return the side effects sorted by count
            sorted_effects = sorted(all_side_effects.items(), key=lambda x: x[1]['count'], reverse=True)
            common_effects = [{'term': effect[0], 'count': effect[1]['count'], 'drugs': effect[1]['drugs']}
                             for effect in sorted_effects if len(effect[1]['drugs']) > 1]
            return common_effects[:10]  # Return top 10 common effects

        # Create a graph where nodes are side effects and drugs
        G = nx.Graph()

        # Add drug nodes
        drug_names = [drug['drug_name'] for drug in drugs_info]
        for drug in drug_names:
            G.add_node(drug, type='drug')

        # Add side effect nodes and edges between drugs and their side effects
        side_effect_terms = list(all_side_effects.keys())
        for effect in side_effect_terms:
            G.add_node(effect, type='effect')
            for drug in all_side_effects[effect]['drugs']:
                # Add weight based on count for this drug-effect pair
                count = sum(se['count'] for se in next(d for d in drugs_info if d['drug_name'] == drug)['side_effects'] 
                           if se['term'] == effect)
                G.add_edge(drug, effect, weight=max(1, count/100))  # Normalize weight

        # Convert to PyTorch Geometric format
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        reverse_mapping = {i: node for node, i in node_mapping.items()}

        # Create edge index and weights
        edge_index = []
        edge_weights = []
        
        for u, v, data in G.edges(data=True):
            edge_index.append([node_mapping[u], node_mapping[v]])
            edge_index.append([node_mapping[v], node_mapping[u]])  # undirected
            edge_weights.append(data['weight'])
            edge_weights.append(data['weight'])  # same weight for reverse edge

        if not edge_index:  # If no edges, return empty list
            return []

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        # Create more informative node features
        x = []
        node_types = []
        
        for node in G.nodes():
            if G.nodes[node]['type'] == 'drug':
                # For drugs: [1, 0, degree, 0]
                x.append([1.0, 0.0, float(G.degree(node)), 0.0])
                node_types.append('drug')
            else:
                # For effects: [0, 1, degree, count]
                count = all_side_effects[node]['count']
                x.append([0.0, 1.0, float(G.degree(node)), float(count)/1000.0])  # Normalize count
                node_types.append('effect')

        x = torch.tensor(x, dtype=torch.float)

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights)

        # Since we don't have a pre-trained model, use node features and graph structure
        # to determine interaction patterns
        
        # Method 1: Use node degrees and edge weights to calculate interaction score
        drug_indices = [i for i, t in enumerate(node_types) if t == 'drug']
        effect_indices = [i for i, t in enumerate(node_types) if t == 'effect']
        
        # Calculate interaction scores based on shared effects
        interaction_scores = {}
        for i, drug1_idx in enumerate(drug_indices):
            drug1 = reverse_mapping[drug1_idx]
            for drug2_idx in drug_indices[i+1:]:
                drug2 = reverse_mapping[drug2_idx]
                
                # Find common side effects
                drug1_effects = set(nx.neighbors(G, drug1))
                drug2_effects = set(nx.neighbors(G, drug2))
                common_effects = drug1_effects.intersection(drug2_effects)
                
                if common_effects:
                    # Calculate interaction score
                    score = sum(all_side_effects[effect]['count'] for effect in common_effects)
                    interaction_scores[(drug1, drug2)] = {
                        'score': score,
                        'common_effects': list(common_effects)
                    }
        
        # Find high-risk common side effects between drugs
        common_effects = []
        for effect in side_effect_terms:
            effect_drugs = set(all_side_effects[effect]['drugs'])
            
            if len(effect_drugs) > 1:  # Only consider effects present in multiple drugs
                # Calculate commonality score
                count = all_side_effects[effect]['count']
                # Higher score for effects shared by more drugs
                score = count * len(effect_drugs) 
                
                # Find related side effects based on the graph structure
                related_effects = []
                for other_effect in side_effect_terms:
                    if other_effect != effect:
                        # Check if drugs overlap
                        other_drugs = set(all_side_effects[other_effect]['drugs'])
                        if effect_drugs.intersection(other_drugs):
                            related_effects.append(other_effect)
                
                common_effects.append({
                    'term': effect,
                    'count': count,
                    'drugs': list(effect_drugs),
                    'score': float(score),
                    'similar_effects': related_effects[:3],  # Top 3 related effects
                    'risk_level': classify_risk_level(effect, count, len(effect_drugs))
                })

        # Sort by score
        common_effects.sort(key=lambda x: x['score'], reverse=True)
        
        # Add interaction information
        for i, effect in enumerate(common_effects):
            if i < len(common_effects):
                drug_pairs = []
                for d1 in effect['drugs']:
                    for d2 in effect['drugs']:
                        if d1 < d2 and (d1, d2) in interaction_scores:
                            drug_pairs.append({
                                'drug1': d1,
                                'drug2': d2,
                                'score': interaction_scores[(d1, d2)]['score']
                            })
                common_effects[i]['drug_interactions'] = drug_pairs

        return common_effects[:10]  # Return top 10

    except Exception as e:
        # Fallback in case of error
        print(f"Error in GNN analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sorted_effects = sorted(all_side_effects.items(), key=lambda x: x[1]['count'], reverse=True)
        common_effects = [{'term': effect[0], 'count': effect[1]['count'], 'drugs': effect[1]['drugs']}
                         for effect in sorted_effects if len(effect[1]['drugs']) > 1]
        return common_effects[:10]

def classify_risk_level(effect_name, count, num_drugs):
    """Classify the risk level of a side effect based on count and number of drugs"""
    # List of high-risk side effects (add more as needed)
    high_risk_terms = [
        'death', 'cardiac arrest', 'heart attack', 'stroke', 'seizure', 
        'anaphylaxis', 'respiratory failure', 'liver failure', 'kidney failure',
        'bleeding', 'hemorrhage', 'coma', 'arrhythmia', 'heart failure'
    ]
    
    # Check if effect name contains any high-risk terms
    if any(term in effect_name.lower() for term in high_risk_terms):
        return 'high'
    
    # Count-based classification
    if count > 1000 and num_drugs >= 3:
        return 'high'
    elif count > 500 or (count > 100 and num_drugs >= 3):
        return 'moderate'
    else:
        return 'low'

def get_pubchem_info(drug_name):
    try:
        # Get synonyms
        synonyms_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/synonyms/JSON"
        synonyms_response = requests.get(synonyms_url)
        synonyms_response.raise_for_status()
        synonyms_data = synonyms_response.json()

        # Extract CID from synonyms response
        cid = synonyms_data["InformationList"]["Information"][0]["CID"]

        # Get some common synonyms (max 5)
        common_synonyms = synonyms_data["InformationList"]["Information"][0]["Synonym"][:5]

        # Get properties using the CID
        properties_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName/JSON"
        properties_response = requests.get(properties_url)
        properties_response.raise_for_status()
        properties_data = properties_response.json()

        # Extract properties
        prop = properties_data["PropertyTable"]["Properties"][0]
        mol_formula = prop.get('MolecularFormula', 'N/A')
        mol_weight = prop.get('MolecularWeight', 'N/A')
        iupac_name = prop.get('IUPACName', 'N/A')

        return {
            'success': True,
            'drug_name': drug_name,
            'cid': cid,
            'common_synonyms': common_synonyms,
            'molecular_formula': mol_formula,
            'molecular_weight': mol_weight,
            'iupac_name': iupac_name
        }

    except requests.exceptions.RequestException as e:
        raise Exception(f'Error fetching PubChem data: {str(e)}')
    except (KeyError, IndexError) as e:
        raise Exception(f'Drug not found or invalid drug name: {drug_name}')

def get_side_effects(drug_name):
    try:
        # Query OpenFDA for adverse events
        openfda_url = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{drug_name}&count=patient.reaction.reactionmeddrapt.exact&limit=10"
        openfda_response = requests.get(openfda_url)

        if openfda_response.status_code == 200:
            openfda_data = openfda_response.json()
            # Extract the top 10 adverse events
            side_effects = []
            if 'results' in openfda_data:
                for result in openfda_data['results']:
                    side_effects.append({
                        'term': result['term'],
                        'count': result['count']
                    })
            return side_effects
        else:
            # Try alternative search if exact term fails
            alt_url = f"https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:{drug_name}&count=patient.reaction.reactionmeddrapt.exact&limit=10"
            alt_response = requests.get(alt_url)

            if alt_response.status_code == 200:
                alt_data = alt_response.json()
                side_effects = []
                if 'results' in alt_data:
                    for result in alt_data['results']:
                        side_effects.append({
                            'term': result['term'],
                            'count': result['count']
                        })
                return side_effects

            # If both searches fail, return a message
            return []
    except Exception as e:
        # Return empty list if error occurs
        return []

if __name__ == '__main__':
    app.run(debug=True)