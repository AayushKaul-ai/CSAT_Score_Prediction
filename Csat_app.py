from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

class FrequencyEncoder:
    def __init__(self, columns):
        self.columns = columns
        self.maps_ = {}

    def fit(self, X, y=None):
        import pandas as pd
        X_ = pd.DataFrame(X).copy()
        X_.columns = self.columns
        for c in self.columns:
            freqs = X_[c].value_counts(dropna=False)
            total = freqs.sum()
            self.maps_[c] = (freqs / total).to_dict()
        return self

    def transform(self, X):
        import pandas as pd
        X_ = pd.DataFrame(X).copy()
        X_.columns = self.columns
        for c in self.columns:
            X_[c] = X_[c].map(self.maps_.get(c, {})).fillna(0.0).astype(float)
        return X_.values

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Load trained model
model = joblib.load("model.pkl")

# Extract expected input fields
try:
    expected_fields = list(model.feature_names_in_)
except:
    expected_fields = None

@app.route("/", methods=["GET"])
def home():
    return "<h2>✅ CSAT Prediction API is running successfully!</h2>"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    if expected_fields:
        df = df.reindex(columns=expected_fields, fill_value=None)

    pred = model.predict(df)
    pred = pred + 1  # Convert label back to 1-5

    return jsonify({"predicted_CSAT": int(pred[0])})

if __name__ == "__main__":
    app.run(debug=True)