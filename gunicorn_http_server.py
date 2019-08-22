# -*- coding: utf-8 -*-

import time
from flask import Flask, request


app = Flask(__file__)


@app.route('/', methods=['POST'])
def server_api():
    data = request.data
    time.sleep(1)
    return "Hello World!"


if __name__ == '__main__':
    app.run()
