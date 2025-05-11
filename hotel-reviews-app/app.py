from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Load top model & vectorizer
model = load("model.joblib")
vectorizer = load("vectorizer.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    confidence = None
    if request.method == "POST":
        review = request.form["review"]
        if review.strip():
            vectorized = vectorizer.transform([review])
            prediction = model.predict(vectorized)[0]
            prob_accuracy = model.predict_proba(vectorized)[0]
            confidence = round(prob_accuracy[prediction] * 100,2)
            sentiment = "Positive 😊" if prediction == 1 else "Negative 😢"
    return render_template("index.html", sentiment=sentiment, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
