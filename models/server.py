import numpy as np
import pickle
import json
import requests
import logging
import time
import h5py
import os


def ask_server_for_inception_score(filename, port=5000):
    server = 'localhost'
    with open("server_for_inception_score.config") as f:
        server = f.readline()
        server = server.rstrip()
    json_data = {'filename': filename}
    start = time.time()
    headers = {'Content-Type': 'application/json'}
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    r = requests.post("http://{}:{}/inception-score".format(server, port),
                      headers=headers,
                      data=json.dumps(json_data))
    print("[Remote IS] Total request took {}s".format(time.time() - start))
    response_data = json.loads(r.text)
    print("[Remote IS] Computation took {}s".format(response_data['computation_time']))
    return float(response_data['mean']), float(response_data['std'])
