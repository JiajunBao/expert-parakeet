import numpy as np

from flask import Flask
from flask import render_template
from flask import request
from flask import Response
from flask import jsonify

import numpy
from src.deploy import get_activation
# from classify import classify


app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/activations')
def activations():
    """
    Receive a text and return HATTN activation map
    """
    if request.method == 'GET':
        text = request.args.get('text', '')
        if len(text.strip()) == 0:
            return Response(status=400)

        data = get_activation(text)
        # data = {
        #     'activations': [[[0.1, 0.1, 0.2, 0.6], 0.3], [[0.2, 0.3, 0.5], 0.6]],
        #     'doc': [["how", "are", "you", "buddy"], ["I", "am", "fine"]],
        #     'scores': [0.3, 0.7],
        #     'categories': ["label1", "label2"]
        # }
        return jsonify(data)
    else:
        return Response(status=501)
