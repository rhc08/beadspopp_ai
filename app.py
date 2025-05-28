from flask import Flask, request, jsonify, send_from_directory
import time
from palette import extract_palette

app = Flask(__name__, static_folder='static')

@app.route('/extract', methods=['POST'])
def extract():
    img_bytes = request.files['image'].read()
    palette   = extract_palette(img_bytes)
    return jsonify({
        'id': str(int(time.time()*1000)),
        'palette': palette
    })

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


