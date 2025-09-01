HOCKEY-MIND
HOCKEY-MIND is a specialized AI-powered application designed to assist field hockey coaches, players, and enthusiasts with tailored answers and YouTube video recommendations for training, drills, strategies, rules, and tutorials. Built with FastAPI, it leverages a SQLite database, FAISS for semantic search, and OpenRouter's GPT-4o model for natural language processing. The system supports both English and Dutch queries, with a focus on field hockey content, and provides up to five relevant video recommendations based on semantic similarity.
Table of Contents

Project Overview
Features
Requirements
Installation
Configuration
Database Setup
Running the Application
API Endpoints
Usage Examples
Project Structure
Troubleshooting
Contributing
License

Project Overview
HOCKEY-MIND is a backend API service that processes field hockey-related queries, provides concise and practical answers, and recommends relevant YouTube videos from a pre-populated SQLite database. It uses the sentence-transformers library for embedding generation, FAISS for efficient similarity search, and Google Translate for multilingual support (English and Dutch). The system is designed to filter out non-field hockey queries (e.g., ice hockey, politics) and focus on relevant topics like backhand shooting, penalty corners, and youth training drills.
Key components:

FastAPI: Provides a robust and asynchronous API framework.
SQLite Database: Stores YouTube video metadata (title, URL, embeddings).
FAISS: Enables fast semantic search for video recommendations.
OpenRouter (GPT-4o): Generates accurate and context-aware responses.
Multilingual Support: Handles English and Dutch queries with translation and hockey-specific terminology.

Features

Field Hockey Q&A: Answers questions about training, drills, strategies, rules, and tutorials, tailored to user roles (e.g., coach) and teams (e.g., U8C).
Video Recommendations: Returns up to five YouTube videos with the highest semantic similarity to the query (similarity > 0.3).
Multilingual Support: Processes queries in English and Dutch, with automatic translation and a hockey-specific translation dictionary.
Conversation History: Maintains the last three interactions per user-team combination for context-aware responses.
Domain Filtering: Rejects out-of-domain queries (e.g., ice hockey, politics) with polite refusal messages.
Semantic Search: Uses FAISS and paraphrase-multilingual-MiniLM-L12-v2 for accurate video recommendations.
Error Handling: Includes retry logic for API calls and robust logging for debugging.

Requirements

Python: Version 3.10 or higher.
Dependencies: Listed in requirements.txt (see below).
OpenRouter API Key: Required for AI responses.
SQLite Database: A pre-populated HockeyFood.db with a YouTube_Urls table.
System: Compatible with macOS, Linux, or Windows.

Dependencies (from requirements.txt):

fastapi==0.115.14
uvicorn==0.35.0
python-dotenv==1.1.1
pydantic>=2,<3
langdetect==1.0.9
deep-translator==1.11.4
beautifulsoup4==4.13.4
requests==2.32.4
sentence-transformers>=3.0.0
faiss-cpu==1.8.0
Additional dependencies (e.g., torch, transformers, numpy) as specified in requirements.txt.

Installation

Clone the Repository:
git clone https://github.com/your-username/hockey-mind.git
cd hockey-mind


Set Up a Virtual Environment:
python -m venv hockey-recommender
source hockey-recommender/bin/activate  # On Windows: hockey-recommender\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Verify Dependencies:Ensure all packages are installed correctly:
pip list



Configuration

Create a .env File:In the project root, create a .env file with the following:
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
DATABASE_PATH=HockeyFood.db

Replace sk-or-v1-your-api-key-here with your OpenRouter API key. The DATABASE_PATH defaults to HockeyFood.db if not specified.

Verify Environment Variables:Run the following to check if the API key is loaded:
from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv("OPENROUTER_API_KEY"))



Database Setup
The application requires a SQLite database (HockeyFood.db) with a YouTube_Urls table containing video metadata and embeddings. The table schema is:
CREATE TABLE YouTube_Urls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    embedding BLOB
);

Populating the Database

Check Existing Data:
sqlite3 HockeyFood.db "SELECT title, url FROM YouTube_Urls LIMIT 5"


Add Sample Data (if needed):Use the following script to populate the database with field hockey videos:
import sqlite3
import pickle
from sentence_transformers import SentenceTransformer

conn = sqlite3.connect('HockeyFood.db')
cursor = conn.cursor()
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

videos = [
    ("Field Hockey Backhand Shooting Drills for Kids", "https://youtube.com/example1"),
    ("Veldhockey Backhandschoten Oefeningen", "https://youtube.com/example2"),
    ("Youth Field Hockey Skills", "https://youtube.com/example3"),
    ("Beginner Veldhockey Training", "https://youtube.com/example4"),
    ("Field Hockey Drills for Backhand Accuracy", "https://youtube.com/example5"),
    ("Advanced Penalty Corner Techniques", "https://youtube.com/example6")
]

for title, url in videos:
    embedding = model.encode(title, convert_to_tensor=False)
    cursor.execute("INSERT INTO YouTube_Urls (title, url, embedding) VALUES (?, ?, ?)",
                  (title, url, pickle.dumps(embedding)))
conn.commit()
conn.close()


Generate Embeddings:After adding videos, ensure embeddings are saved to video_embeddings.npy and metadata to video_metadata.json by running OpenAPI_DB.py once:
python OpenAPI_DB.py



Running the Application

Start the FastAPI Server:
uvicorn app:app --host 0.0.0.0 --port 8000

The API will be available at http://localhost:8000.

Verify the API:Open a browser or use curl to check the API root:
curl http://localhost:8000

You should see a JSON response with the API title and version.


API Endpoints
The API provides two endpoints for submitting field hockey queries:
1. GET /ask

Description: Submits a query via query parameters and returns an AI response with up to five YouTube video recommendations.
Parameters:
user_active_role (str): The user's role (e.g., "le Coach").
user_team (str): The user's team (e.g., "U8C").
user_prompt (str): The field hockey-related question (e.g., "Geef me oefeningen voor backhandschoten").


Example:curl "http://localhost:8000/ask?user_active_role=le%20Coach&user_team=U8C&user_prompt=Geef%20me%20oefeningen%20voor%20backhandschoten"



2. POST /ask

Description: Submits a query via a JSON body and returns an AI response with up to five YouTube video recommendations.
Request Body:{
  "user_active_role": "le Coach",
  "user_team": "U8C",
  "user_prompt": "Geef me oefeningen voor backhandschoten"
}


Example:curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"user_active_role":"le Coach","user_team":"U8C","user_prompt":"Geef me oefeningen voor backhandschoten"}'



Response Format
Both endpoints return:
{
  "ai_response": "Voor je U8C-team zijn de beste oefeningen voor backhandschoten simpel en leuk. 1. Kegelschietoefeningen: Zet kegels op en laat kinderen richten op specifieke doelen om precisie te verbeteren. 2. Polsbewegingsoefening: Oefen het 'flicken' van de pols om kracht te ontwikkelen. 3. Partneroefening: Laat kinderen in paren backhandpasses geven om controle te krijgen. Gebruik lichte ballen voor veiligheid.",
  "recommended_content_details": [
    {"title": "Field Hockey Backhand Shooting Drills for Kids", "url": "https://youtube.com/example1", "similarity": 0.823},
    {"title": "Veldhockey Backhandschoten Oefeningen", "url": "https://youtube.com/example2", "similarity": 0.815},
    {"title": "Youth Field Hockey Skills", "url": "https://youtube.com/example3", "similarity": 0.792},
    {"title": "Beginner Veldhockey Training", "url": "https://youtube.com/example4", "similarity": 0.780},
    {"title": "Field Hockey Drills for Backhand Accuracy", "url": "https://youtube.com/example5", "similarity": 0.765}
  ]
}

Usage Examples

Ask About Backhand Drills (Dutch):
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"user_active_role":"le Coach","user_team":"U8C","user_prompt":"Geef me oefeningen voor backhandschoten voor kinderen"}'


Ask About Penalty Corners (English):
curl "http://localhost:8000/ask?user_active_role=le%20Coach&user_team=U8C&user_prompt=How%20to%20train%20for%20penalty%20corners"


Test with Python Client:
import httpx
import asyncio

async def test_query():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/ask",
            json={
                "user_active_role": "le Coach",
                "user_team": "U8C",
                "user_prompt": "Geef me de beste hockeyoefeningen voor kinderen om hun backhandschoten te verbeteren"
            }
        )
        print(response.json())

asyncio.run(test_query())



Project Structure
hockey-mind/
├── OpenAPI_DB.py        # Core logic for query processing and FAISS search
├── app.py               # FastAPI application with API endpoints
├── requirements.txt     # Project dependencies
├── requirements.in      # Dependency input file for pip-compile
├── .env                 # Environment variables (API key, database path)
├── HockeyFood.db        # SQLite database with YouTube_Urls table
├── video_embeddings.npy # Precomputed FAISS embeddings
├── video_metadata.json  # Video metadata (title, URL)

Troubleshooting

API Key Error:

Symptom: OPENROUTER_API_KEY not set in .env file.
Fix: Ensure .env contains a valid OpenRouter API key:OPENROUTER_API_KEY=sk-or-v1-your-api-key-here




Database Not Found:

Symptom: Database file not found at HockeyFood.db.
Fix: Verify HockeyFood.db exists in the project root or update DATABASE_PATH in .env.


No Video Recommendations:

Symptom: recommended_content_details is empty.
Fix:
Check HockeyFood.db for relevant entries:sqlite3 HockeyFood.db "SELECT title FROM YouTube_Urls WHERE title LIKE '%field hockey%' LIMIT 5"


Add more field hockey videos using the database population script above.
Regenerate embeddings:rm video_embeddings.npy video_metadata.json
python OpenAPI_DB.py






Fewer than 5 Recommendations:

Symptom: Less than 5 videos in recommended_content_details.
Cause: Database may have fewer than 5 entries with similarity > 0.3 or matching field hockey criteria.
Fix: Add more relevant videos to YouTube_Urls table.


Translation Errors:

Symptom: Logs show Translation error from deep-translator.
Fix: Ensure internet connectivity and test Google Translate:from deep_translator import GoogleTranslator
print(GoogleTranslator(source="nl", target="en").translate("veld hockey"))




Check Logs:

Logs are written to the console or can be redirected:uvicorn app:app --host 0.0.0.0 --port 8000 > debug.log 2>&1


Look for errors like FAISS search error or OpenRouter API error.



Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please include tests and update the README if new features are added.
License
This project is licensed under the MIT License. See the LICENSE file for details.
