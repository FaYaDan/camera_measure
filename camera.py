from flask import Flask, jsonify
from subprocess import Popen

app = Flask(__name__)
@app.route('/run_capstone')
def run_capstone():
    p = Popen(['python', 'Capstone.py'])
    return jsonify({'message': 'Capstone script is running.'})
if __name__ == '__main__':
    app.run()
