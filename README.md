# news-summarization-and-text-to-speech
Overview

This application provides company-related news analysis, including summarization, sentiment analysis, comparative analysis, and text-to-speech conversion.

Installation & Setup

#Install Dependencies

#Backend Setup
Navigate to the backend directory:

cd backend
pip install -r requirements.txt

#Frontend Setup
Navigate to the frontend directory:

cd frontend
pip install -r requirements.txt

#Run the Application

Start the Backend Server

cd backend
python app.py

Start the Frontend Application

cd frontend
streamlit run app.py

#API Usage
#Available Endpoint

Fetch Company Data

Endpoint: /get_company_data

Method: POST

Request Body:
              {
                "company": "Tesla"
              }
{
  "news": [...],
  "frequency": {...},
  "comparative_analysis": {...},
  "final_sentiment": "Overall, Tesla has received positive sentiment in recent news.",
  "audio_url": "http://localhost:5000/static/sentiment_Tesla.mp3"
}

#Testing the API with Postman

Open Postman.
Set the request method to POST.
Enter the API URL: http://localhost:7650/get_company_data
Go to Body > raw and select JSON format.
Paste the request body: {"company": "Tesla"}
Click Send.
