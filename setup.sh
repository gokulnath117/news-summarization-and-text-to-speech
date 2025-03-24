#!/bin/bash
# Start Flask API in the background
cd backend && python app.py &

# Start Streamlit frontend
cd ../frontend && streamlit run app.py
