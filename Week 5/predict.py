"""
Let's predict customer churn through web-service
"""
import pickle
from flask import Flask, request, jsonify

dv_file = 'dv.bin'
model_file = 'model1.bin'
# model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_on:
    dv = pickle.load(f_on)

app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
