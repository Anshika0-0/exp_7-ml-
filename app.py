import pickle           

# # hardcore input values for testing
# test_input = [[2.1, 2.5, 1.4, 3.2]]  # Example input for prediction
# prediction = iris_model.predict(test_input)
# print("Prediction for the test input:", prediction)


from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)

with open('iris_model.pkl', 'rb') as file:
    iris_model = pickle.load(file)

@app.route ("/")         #write endpoint inside it " "
def home():
    return jsonify({"message":"app is running..."}) # key : value pair in json format
@app.route("/health")
def health():
    return jsonify({"status":"app is healthy..."})

@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
    s_l = data["sepal_length"]
    p_l = data["petal_length"]
    s_w = data["sepal_width"]
    p_w = data["petal_width"]

    pred = iris_model.predict([[s_l, s_w, p_l, p_w]])
    return jsonify({"prediction": int(pred[0])})  # Convert prediction to int for JSON serialization    

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)