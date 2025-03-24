import os
from flask import Flask, request, jsonify,send_file, url_for
from flask_cors import CORS
from utils import get_unique_news, sentiment_distribution, generate_comparative_analysis, generate_final_sentiment, english_to_hindi_tts

app = Flask(__name__)
CORS(app) 

@app.route('/get_company_data', methods=['POST'])
def get_company_data():
    company = request.json.get('company')
    news = get_unique_news(company)
    frequency = sentiment_distribution(news)
    comparative_analysis = generate_comparative_analysis(news)
    final_sentiment = generate_final_sentiment(news)

    english_text=final_sentiment["Final Sentiment Analysis"]
    audio_buffer=english_to_hindi_tts(english_text)
    # Save audio to a file
    os.makedirs("static", exist_ok=True)
    audio_filename = f"sentiment_{company}.mp3"
    audio_path = os.path.join("static", audio_filename)

    with open(audio_path, "wb") as f:
        f.write(audio_buffer.read())

    # Generate URL for audio file
    audio_url = url_for("static", filename=audio_filename, _external=True)

    if not news:
        return jsonify({"error": "Enter the Company name correctly"}), 400
    return jsonify({
        "news": news,
        "frequency": frequency,
        "comparative_analysis": comparative_analysis,
        "final_sentiment": final_sentiment,
        "audio_url": audio_url
    })


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)