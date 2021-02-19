#!/usr/bin/env python3

import connexion
from api.server_inference import infer


def predict():
    row = connexion.request.form['row'].split(',')
    price = infer(row)
    return {'price': price}


if __name__ == '__main__':
    app = connexion.FlaskApp(__name__, port=5090, specification_dir='swagger/')
    app.add_api('swagger.yaml')
    app.run()
