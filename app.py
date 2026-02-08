from flask import Flask, request, jsonify
from flask_cors import CORS
from scout_engine import ScoutEngine
import os

app = Flask(__name__)
CORS(app)  # Allows your website to talk to this server

# Initialize Engine
DATA_FILE = 'scouting_with_market_value2024-2025.csv' # Developer must put CSV here
engine = ScoutEngine(DATA_FILE)

print("⏳ Initializing Scouting Engine... (Training Models)")
if engine.load_data():
    print("✅ Engine Ready!")
else:
    print("❌ Failed to load data. Check file path.")

@app.route('/api/config', methods=['GET'])
def get_config():
    """Returns presets, labels, and player list for the UI."""
    config = engine.get_config()
    config['players'] = engine.get_player_list()
    return jsonify(config)

@app.route('/api/scout', methods=['POST'])
def scout_players():
    """
    Input JSON:
    {
        "role": "Striker (Poacher)", 
        "budget": 50, 
        "max_age": 28, 
        "weights": {"Gls_p90": 10, "npxG_p90": 8} 
    }
    """
    data = request.json
    results = engine.attribute_search(
        weights=data.get('weights', {}),
        role=data.get('role', 'All'),
        budget=data.get('budget', 100),
        max_age=data.get('max_age', 40)
    )
    return jsonify(results)

@app.route('/api/clone', methods=['GET'])
def clone_player():
    """Usage: /api/clone?name=Erling Haaland"""
    name = request.args.get('name')
    if not name:
        return jsonify({"error": "Missing name parameter"}), 400
        
    results = engine.find_clones(name)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)