"""
Vercel Python Serverless Function
GET /api/metadata
Returns the dropdown options / numeric ranges the frontend form needs,
generated from the training data by train_model.py.
"""

import json
import os
from http.server import BaseHTTPRequestHandler

METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metadata.json")

with open(METADATA_PATH) as f:
    _metadata = json.load(f)


class handler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        self._set_headers(200)
        self.wfile.write(json.dumps(_metadata).encode())
