from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("jarvisx17/japanese-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("jarvisx17/japanese-sentiment-analysis")

# Create the Flask app
app = Flask(__name__)


# Define the endpoint for sentiment analysis
@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    # Get the text from the request body
    text = request.json['text']

    # Tokenize the text and pass it through the model
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)

    # Get the predicted sentiment label and score
    label = outputs.logits.argmax().item()
    score = outputs.logits.softmax(dim=1).tolist()[0]

    # Return the result as a JSON object
    return jsonify({'label': label, 'score': score})


# Run the app
if __name__ == '__main__':
    app.run()
