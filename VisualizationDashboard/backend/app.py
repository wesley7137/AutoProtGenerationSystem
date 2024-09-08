from flask import Flask, jsonify, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
import os
import time
import json

app = Flask(__name__)
CORS(app)

# Simulated data
sequences = pd.DataFrame({
    'id': range(1, 101),
    'sequence': [''.join(np.random.choice(['A', 'C', 'G', 'T'], 10)) for _ in range(100)],
    'score': np.random.rand(100),
    'status': np.random.choice(['Generated', 'Optimized', 'Analyzed'], 100)
})

@app.route('/api/sequences')
def get_sequences():
    return jsonify(sequences.to_dict('records'))

@app.route('/api/sequences/<int:id>')
def get_sequence(id):
    sequence = sequences[sequences['id'] == id].to_dict('records')[0]
    # Simulate PDB file generation
    pdb_file = f"sequence_{id}.pdb"
    with open(pdb_file, 'w') as f:
        f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N")
    sequence['pdbUrl'] = f"/static/{pdb_file}"
    return jsonify(sequence)

@app.route('/api/events')
def events():
    def generate():
        while True:
            # Simulate event generation
            event = {
                'message': f'New sequence optimized: {np.random.randint(1, 101)}',
                'severity': np.random.choice(['info', 'success', 'warning', 'error'])
            }
            yield f"data: {json.dumps(event)}\n\n"
            time.sleep(10)  # Send an event every 10 seconds
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)