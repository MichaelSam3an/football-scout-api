from flask import Flask, request, jsonify
from flask_cors import CORS
from scout_engine import ScoutEngine
import os
import traceback

app = Flask(__name__)
CORS(app)  # Allows your website to talk to this server

# Safe CSV path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "scouting_with_market_value2024-2025.csv")

# Initialize Engine
engine = ScoutEngine(DATA_FILE)

print("⏳ Initializing Scouting Engine... (Training Models)")
try:
    engine.load_data()
    engine.run_pipeline()
    print("✅ Engine Ready!")
except Exception as e:
    print("❌ Failed to initialize engine:")
    print(e)
    traceback.print_exc()


@app.route("/api/config", methods=["GET"])
def get_config():
    """Returns presets, labels, features, and player list for the UI."""
    try:
        config = engine.get_config()
        config["players"] = engine.get_player_list()
        return jsonify(config)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/scout", methods=["POST"])
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
    try:
        data = request.get_json(silent=True) or {}

        results = engine.attribute_search(
            weights=data.get("weights", {}),
            role=data.get("role", "All"),
            budget=data.get("budget", 100),
            max_age=data.get("max_age", 40),
        )
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/clone", methods=["GET"])
def clone_player():
    """Usage: /api/clone?name=Erling Haaland"""
    try:
        name = request.args.get("name")
        if not name:
            return jsonify({"error": "Missing name parameter"}), 400

        results = engine.find_clones(name)
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
