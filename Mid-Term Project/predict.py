
import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_rf.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('win_prediction')


@app.route('/predict', methods=['POST'])
def predict():
    match = request.get_json()

    X = dv.transform([match])
    y_pred = model.predict_proba(X)[:, 1]
    match_win = y_pred >= 0.5

    result = {
        'win_probability': float(y_pred),
        'match_win': bool(match_win)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
