from flask import Flask, render_template, request, jsonify
import random
import nltk
from transformers import pipeline

# Download necessary NLP data
nltk.download('punkt')

app = Flask(__name__)

# AI Model for NLP (Using Transformers)
nlp_model = pipeline("text-generation", model="gpt2")

# Predefined responses (for simple queries)
responses = {
    "hello": "Hi there! How can I assist you?",
    "how are you": "I'm just a chatbot, but I'm doing great! How about you?",
    "bye": "Goodbye! Have a great day!",
    "who created you": "anushka!",
}

# Function to generate chatbot responses
def chatbot_response(user_input):
    user_input = user_input.lower()

    # Check predefined responses
    for key in responses.keys():
        if key in user_input:
            return responses[key]

    # AI-generated response
    generated_text = nlp_model(user_input, max_length=50, do_sample=True)[0]["generated_text"]
    return generated_text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    bot_reply = chatbot_response(user_message)
    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
