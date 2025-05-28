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

if __name__ == '__main__':
    app.run(debug=True)

