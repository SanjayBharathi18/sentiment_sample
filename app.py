from flask import Flask, request, jsonify
from flask_cors import CORS
from sentiment import process_videos  # make sure your full script is in 'your_script.py'

app = Flask(__name__)
CORS(app)  # 🔥 Enables requests from any frontend like Vercel

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    channel_name = data.get("channel_name")
    if not channel_name:
        return jsonify({"error": "Channel name missing"}), 400
    result = process_videos(channel_name, save_csv=False)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
