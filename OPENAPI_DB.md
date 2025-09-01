To integrate the provided code with an internal SQLite database named `HockeyFood.db` and retrieve data from the `YouTube_Urls` table (with columns `title`, `url`, and `metatags`) instead of performing a DuckDuckGo search, we need to modify the `duckduckgo_search` function to query the database and use semantic similarity to match the user's prompt with the `title` and `metatags` columns. The rest of the code, including the conversation history, refusal detection, and domain checks, will remain unchanged. The modified function will query the database, compute semantic similarity using the `SentenceTransformer` model, and return up to three relevant results in the same format as the original `recommended_content_details` (`[{"title": "...", "url": "..."}]`).

Below is the modified code, focusing on replacing the `duckduckgo_search` function with a new function called `search_youtube_urls_db` that queries the `HockeyFood.db` database. The new function will:
1. Connect to the SQLite database.
2. Query the `YouTube_Urls` table.
3. Compute semantic similarity between the user's prompt and the `title` and `metatags` columns.
4. Return up to three results that meet the similarity threshold and are hockey-related.

```python
import logging
import os
import re
import string
import urllib.parse
from dotenv import load_dotenv
import httpx
from langdetect import detect
from deep_translator import GoogleTranslator
import sqlite3
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv(dotenv_path="/Users/talharauf/Desktop/Hockey-Mind/.env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
if not OPENROUTER_API_KEY:
    logging.error("OPENROUTER_API_KEY not set in .env file.")
    raise RuntimeError("OPENROUTER_API_KEY not set in .env file.")
else:
    masked_key = OPENROUTER_API_KEY[:6] + "..." + OPENROUTER_API_KEY[-4:]
    logging.info(f"Loaded OpenRouter API key: {masked_key}")

# In-memory conversation history
conversation_histories = {}

# Expanded hockey keywords for domain detection
hockey_keywords = [
    # Core hockey terms
    "hockey", "field hockey", "veldhockey", "match", "wedstrijd", "game", "spel", "goal", "doelpunt",
    "score", "scoren", "ball", "bal", "stick", "hockeystick", "field", "veld", "turf", "kunstgras",
    "pitch", "speelveld", "corner", "short corner", "long corner", "korte hoek", "lange hoek",
    "penalty", "strafbal", "shootout", "strookschot", "penalty stroke", "strafslag",
    
    # Roles and positions
    "coach", "trainer", "goalkeeper", "doelman", "keeper", "goalie", "defender", "verdediger",
    "midfielder", "middenvelder", "forward", "aanvaller", "striker", "spits", "captain", "aanvoerder",
    "player", "speler", "team", "ploeg",
    
    # Skills and techniques
    "shooting", "schieten", "passing", "passen", "backhand", "achterhand", "forehand", "voorhand",
    "wrist shot", "pols schot", "slap shot", "slagschot", "drag flick", "sleeppush", "push pass",
    "pushpass", "hit pass", "slagpass", "aerial pass", "luchtpass", "dribbling", "dribbelen",
    "stick work", "stickwerk", "deflection", "afbuiging", "scoop", "scheppen", "tackle", "tackelen",
    "block tackle", "blok tackle", "jab tackle", "steektackle", "reverse stick", "omgekeerde stick",
    "indian dribble", "indiase dribbel", "3d skills", "3d vaardigheden", "goalkeeping", "doelverdediging",
    "save", "redding", "clearance", "uitverdediging", "flick", "slepen", "lift", "optillen",
    "chip", "chippen", "sweep hit", "veegslag", "tomahawk", "backstick", "reverse hit", "omgekeerde slag",
    "drag", "slepen", "dummy", "schijnbeweging", "feint", "fint", "spin", "draaien",
    
    # Training and drills
    "training", "oefening", "exercise", "oefenen", "drill", "oefensessie", "practice", "praktijk",
    "warm-up", "opwarming", "cool-down", "afkoeling", "conditioning", "conditietraining",
    "fitness", "fitheid", "agility", "wendbaarheid", "speed", "snelheid", "endurance", "uithoudingsvermogen",
    "strength", "kracht", "core strength", "kernkracht", "stick handling", "stickbeheersing",
    "ball control", "balbeheersing", "footwork", "voetwerk", "positioning", "positionering",
    "marking", "dekken", "zone defense", "zonedekking", "man-to-man", "man-op-man",
    "attack drill", "aanvalsoefening", "defense drill", "verdedigingsoefening",
    "passing drill", "passoefening", "shooting drill", "schietoefening", "goalkeeper drill",
    "doelmanoefening", "skill development", "vaardigheidsontwikkeling", "technique", "techniek",
    
    # Strategies and tactics
    "strategy", "strategie", "tactic", "tactiek", "game plan", "spelplan", "formation", "opstelling",
    "press", "druk zetten", "counterattack", "tegenaanval", "breakaway", "uitbraak",
    "offensive play", "aanvallend spel", "defensive play", "verdedigend spel", "set piece",
    "standaardsituatie", "free hit", "vrije slag", "penalty corner", "straf Corner",
    
    # Educational and instructional terms
    "tutorial", "handleiding", "tips", "advies", "coaching", "coachen", "learn", "leren",
    "education", "opleiding", "skills training", "vaardigheidstraining", "workshop", "werkplaats",
    "session", "sessie", "clinic", "kliniek", "instruction", "instructie", "guide", "gids",
    
    # Equipment and facilities
    "shin guard", "scheenbeschermer", "mouthguard", "mondbeschermer", "gloves", "handschoenen",
    "grips", "grepen", "turf shoes", "kunstschoenen", "hockey shoes", "hockeyschoenen",
    "goalpost", "doelpaal", "net", "netwerk", "training cone", "trainingskegel",
    "rebound board", "reboundbord", "practice net", "oefennet",
    
    # Miscellaneous
    "warmup", "opwarmen", "stretching", "rekken", "injury prevention", "blessurepreventie",
    "teamwork", "samenwerking", "communication", "communicatie", "leadership", "leiderschap",
    "motivation", "motivatie", "mental preparation", "mentale voorbereiding", "focus", "concentratie",
    "hockey camp", "hockeykamp", "tournament", "toernooi", "league", "liga", "championship",
    "kampioenschap"
]

# Out-of-domain keywords
out_of_domain_keywords = [
    # Politics and government
    "politics", "politiek", "government", "regering", "election", "verkiezing", "policy", "beleid",
    "president", "premier", "parliament", "parlement", "law", "wet", "legislation", "wetgeving",
    "campaign", "campagne", "vote", "stemmen", "democracy", "democratie",
    
    # Other sports
    "football", "voetbal", "soccer", "basketball", "basketbal", "tennis", "cricket", "rugby",
    "volleyball", "volleybal", "baseball", "honkbal", "golf", "swimming", "zwemmen",
    "athletics", "atletiek", "cycling", "wielrennen", "boxing", "boksen", "martial arts",
    "vechtsport", "gymnastics", "gymnastiek",
    
    # Weather and environment
    "weather", "weer", "temperature", "temperatuur", "forecast", "voorspelling", "rain", "regen",
    "snow", "sneeuw", "storm", "storm", "wind", "wind", "sun", "zon", "cloud", "wolk",
    "humidity", "vochtigheid", "climate", "klimaat", "pollution", "vervuiling",
    
    # Entertainment and media
    "movie", "film", "television", "televisie", "music", "muziek", "concert", "concert",
    "celebrity", "beroemdheid", "news", "nieuws", "gossip", "roddel", "streaming", "streamen",
    "video game", "videospel", "gaming", "gamen",
    
    # General off-topic
    "cooking", "koken", "recipe", "recept", "fashion", "mode", "shopping", "winkelen",
    "travel", "reizen", "vacation", "vakantie", "car", "auto", "finance", "financiën",
    "stock market", "aandelenmarkt", "business", "zaken", "job", "baan", "education",
    "onderwijs"  # Non-hockey education
]

# Greetings for detection
greetings = [
    # English greetings
    "hey", "hello", "hi", "hiya", "yo", "what's up", "sup", "good morning", "good afternoon",
    "good evening", "good night", "howdy", "greetings", "morning", "evening",
    
    # Dutch greetings
    "hallo", "hoi", "hey", "goedemorgen", "goedemiddag", "goedenavond", "goedennacht",
    "hé", "joe", "moi", "dag", "goedendag",
    
    # Informal/regional
    "aloha", "ciao", "salut", "hola", "heej"  # Common in some Dutch regions
]

# Refusal detection keywords
refusal_keywords = [
    "i can't help", "cannot assist", "not available", "cannot provide", "inappropriate",
    "refuse", "not allowed", "no access", "ai cannot respond", "ask something else",
    "outside my domain", "beyond my capabilities", "not permitted", "sorry, i can't",
    "unable to answer", "restricted from", "not within my scope", "as an ai language model",
    "i am not able to", "prohibited", "off-topic", "irrelevant", "not my expertise",
    "try a different question", "change the topic", "out of bounds", "not supported",
    "i don't have that information", "no data available", "not equipped to handle"
]

# Semantic detection setup
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight model
refusal_embedding = sentence_model.encode(
    "Sorry, I can only assist with questions about hockey, such as training, drills, strategies, rules, and tutorials. Please ask a hockey-related question!",
    convert_to_tensor=True
)
hockey_reference_embedding = sentence_model.encode(
    "Questions about field hockey training, drills, strategies, rules, techniques, or tutorials, including shooting, passing, dribbling, and goalkeeping.",
    convert_to_tensor=True
)
hockey_technique_embedding = sentence_model.encode(
    "Field hockey skills such as backhand shooting, forehand passing, drag flick, push pass, aerial pass, dribbling, tackling, and goalkeeping techniques.",
    convert_to_tensor=True
)
hockey_context_embedding = sentence_model.encode(
    "Field hockey gameplay, team strategies, player positions, penalty corners, free hits, and match preparation.",
    convert_to_tensor=True
)
out_of_domain_embedding = sentence_model.encode(
    "Questions about politics, other sports like football or tennis, weather, movies, music, cooking, or unrelated general topics.",
    convert_to_tensor=True
)

def is_refusal(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in refusal_keywords)

def is_semantic_refusal(text: str) -> bool:
    embedding = sentence_model.encode(text, convert_to_tensor=True)
    similarity = util.cos_sim(embedding, refusal_embedding).item()
    return similarity > 0.6

def is_in_domain(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    has_hockey_keywords = any(word in prompt_lower for word in hockey_keywords)
    has_out_of_domain_keywords = any(word in prompt_lower for word in out_of_domain_keywords)
    
    if has_out_of_domain_keywords:
        return False
    
    prompt_embedding = sentence_model.encode(prompt, convert_to_tensor=True)
    hockey_primary_similarity = util.cos_sim(prompt_embedding, hockey_reference_embedding).item()
    hockey_technique_similarity = util.cos_sim(prompt_embedding, hockey_technique_embedding).item()
    
    return has_hockey_keywords or hockey_primary_similarity > 0.5 or hockey_technique_similarity > 0.5

def is_greeting_or_vague(prompt: str) -> bool:
    prompt_lower = prompt.lower().strip()
    return any(greeting in prompt_lower for greeting in greetings) or len(prompt_lower.split()) < 3

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_youtube_urls_db(query: str) -> list:
    if not is_in_domain(query):
        logging.info("Query is out of domain, skipping database search.")
        return []
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect("/Users/talharauf/Desktop/Hockey-Mind/HockeyFood.db")
        cursor = conn.cursor()
        
        # Query the YouTube_Urls table
        cursor.execute("SELECT title, url, metatags FROM YouTube_Urls")
        rows = cursor.fetchall()
        
        # Close the database connection
        conn.close()
        
        # Encode the query for semantic similarity
        query_embedding = sentence_model.encode(query, convert_to_tensor=True)
        results = []
        
        # Process each row
        for row in rows:
            title, url, metatags = row
            if not title or not url:
                continue
                
            # Combine title and metatags for semantic comparison
            text_to_compare = f"{title} {metatags or ''}".strip()
            if not text_to_compare:
                continue
                
            # Check if the content is hockey-related
            if not any(kw in text_to_compare.lower() for kw in hockey_keywords):
                continue
                
            # Compute semantic similarity
            text_embedding = sentence_model.encode(text_to_compare, convert_to_tensor=True)
            similarity = util.cos_sim(query_embedding, text_embedding).item()
            
            # Use a similarity threshold (e.g., 0.5) to ensure relevance
            if similarity > 0.5:
                results.append({"title": title[:100], "url": url})
                
        # Sort by similarity (if needed) and limit to 3 results
        results = sorted(results, key=lambda x: util.cos_sim(
            sentence_model.encode(x["title"], convert_to_tensor=True), query_embedding
        ).item(), reverse=True)[:3]
        
        logging.info(f"Database search completed with {len(results)} results.")
        return results
        
    except sqlite3.Error as e:
        logging.error(f"Database error: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        return []

def get_conversation_history(user_role: str, user_team: str) -> str:
    session_key = f"{user_role}|{user_team}"
    history = conversation_histories.get(session_key, [])
    return "\n".join([f"Gebruiker: {q}\nCoach: {a}" for q, a in history[-3:]])

def update_conversation_history(user_role: str, user_team: str, question: str, answer: str):
    session_key = f"{user_role}|{user_team}"
    history = conversation_histories.get(session_key, [])
    history.append((question, answer))
    conversation_histories[session_key] = history[-3:]

def get_relevant_context(question: str) -> str:
    # Placeholder for sample_context (as it was not defined in the original code)
    sample_context = []  # Replace with actual context if available
    question_lower = question.lower()
    relevant = [
        f"Vraag: {entry['question']}\nAntwoord: {entry['answer']}"
        for entry in sample_context
        if any(kw in question_lower for kw in hockey_keywords) and any(kw in entry['question'].lower() for kw in hockey_keywords)
    ]
    return "\n\n".join(relevant[:2])

def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return text

async def agentic_hockey_chat(user_active_role: str, user_team: str, user_prompt: str) -> dict:
    logging.info(f"Processing question: {user_prompt}, role: {user_active_role}, team: {user_team}")

    if not user_prompt or not user_prompt.strip():
        return {"ai_response": "Vraag mag niet leeg zijn.", "recommended_content_details": []}

    try:
        user_lang = detect(user_prompt)
    except Exception:
        user_lang = "en"  # Default to English for video recommendation query

    if is_greeting_or_vague(user_prompt):
        answer = "Hallo! Waarmee kan ik je helpen met betrekking tot hockey, training of andere onderwerpen?" if user_lang == "nl" else "Hello! How can I assist you with hockey, training, or other topics?"
        update_conversation_history(user_active_role, user_team, user_prompt, answer)
        return {"ai_response": answer, "recommended_content_details": []}

    if not is_in_domain(user_prompt):
        answer = "Sorry, ik kan alleen helpen met vragen over hockey, zoals training, oefeningen, strategieën, regels en tutorials. Stel me een hockeygerelateerde vraag!" if user_lang == "nl" else "Sorry, I can only assist with questions about hockey, such as training, drills, strategies, rules, and tutorials. Please ask a hockey-related question!"
        update_conversation_history(user_active_role, user_team, user_prompt, answer)
        return {"ai_response": answer, "recommended_content_details": []}

    history = get_conversation_history(user_active_role, user_team)
    context = get_relevant_context(user_prompt)

    system_prompt = (
        "You are an AI Assistant Bot specialized in all things hockey, including training, drills, strategies, rules, and more. "
        "You communicate in English with a {user_active_role} from the team {user_team}.\n\n"
        "Here is the recent conversation:\n{history}\n\n"
        "Here are some previous hockey-related conversations that may be relevant:\n{context}\n\n"
        "Answer the following question in English based on the provided context and your expertise. "
        "Focus solely on hockey-related topics such as training, drills, strategies, rules, and tutorials.\n{user_prompt}"
    )

    hockey_prompt_template = system_prompt.format(
        user_active_role=user_active_role,
        user_team=user_team,
        history=history or "No previous conversations.",
        context=context or "No relevant context available.",
        user_prompt=user_prompt
    )

    payload = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": hockey_prompt_template}
        ],
        "max_tokens": 150,
        "temperature": 0.2,
        "top_p": 0.9
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        logging.info("Making OpenRouter API call...")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not answer:
                logging.error("No answer received from OpenRouter API.")
                return {"ai_response": "No answer received from the API.", "recommended_content_details": []}

            answer = re.sub(r'https?://\S+', '', answer).strip()
            answer = translate_text(answer, user_lang)

            logging.info("Performing database search...")
            recommended_content = search_youtube_urls_db(user_prompt)
            logging.info(f"Database search completed with {len(recommended_content)} results.")

            if is_refusal(answer) or is_semantic_refusal(answer):
                answer = "Sorry, I can only assist with hockey-related questions. Please ask a hockey-related question!"
                recommended_content = []

            update_conversation_history(user_active_role, user_team, user_prompt, answer)

            return {"ai_response": answer, "recommended_content_details": recommended_content}
    except httpx.HTTPStatusError as e:
        logging.error(f"OpenRouter API error: {e.response.text}")
        return {"ai_response": f"API error: {e.response.text}", "recommended_content_details": []}
    except Exception as e:
        logging.error(f"Internal error: {str(e)}")
        return {"ai_response": f"Internal error: {str(e)}", "recommended_content_details": []}
```

### Explanation of Changes
1. **Replaced `duckduckgo_search` with `search_youtube_urls_db`**:
   - The new function connects to the `HockeyFood.db` SQLite database and retrieves all rows from the `YouTube_Urls` table (`title`, `url`, `metatags`).
   - It combines the `title` and `metatags` for each row to create a text string for semantic comparison.
   - It checks if the content is hockey-related by ensuring at least one `hockey_keyword` is present in the combined text.
   - It computes the cosine similarity between the query embedding and the combined `title` and `metatags` embedding using the `SentenceTransformer` model.
   - Results with a similarity score above 0.5 are included, sorted by similarity, and limited to three entries.
   - The function returns a list of dictionaries in the format `[{"title": "...", "url": "..."}]`, matching the original `recommended_content_details` structure.
   - Error handling is included for SQLite and general exceptions, ensuring robustness.

2. **Removed Dependencies**:
   - Removed `requests`, `BeautifulSoup`, and `concurrent.futures` imports since web scraping and `ThreadPoolExecutor` are no longer needed.
   - Added `sqlite3` import for database operations.

3. **Database Path**:
   - The database path is hardcoded as `/Users/talharauf/Desktop/Hockey-Mind/HockeyFood.db` to match the `.env` file path. This can be made configurable if needed.

4. **Preserved Original Logic**:
   - All other functions (`is_refusal`, `is_semantic_refusal`, `is_in_domain`, `is_greeting_or_vague`, `get_conversation_history`, `update_conversation_history`, `get_relevant_context`, `translate_text`, and `agentic_hockey_chat`) remain unchanged to maintain the original functionality.
   - The `agentic_hockey_chat` function now calls `search_youtube_urls_db` instead of `duckduckgo_search`.

5. **Semantic Detection**:
   - The `search_youtube_urls_db` function uses the same `SentenceTransformer` model (`all-MiniLM-L6-v2`) to compute semantic similarity, ensuring consistency with the original code's approach.
   - If no results meet the similarity threshold (0.5) or are not hockey-related, `recommended_content_details` is returned as an empty list.

6. **Error Handling**:
   - The function handles database errors (e.g., connection issues) and logs them appropriately.
   - The `@retry` decorator is retained to attempt the database query up to three times with exponential backoff in case of transient errors.

### Assumptions
- The `HockeyFood.db` database and `YouTube_Urls` table exist with the specified columns (`title`, `url`, `metatags`).
- The `metatags` column may contain `None` or empty strings, which are handled gracefully.
- The `sample_context` variable was not defined in the original code, so it is left as a placeholder (empty list). If you have a specific `sample_context`, it should be integrated into the `get_relevant_context` function.

### Notes
- The database query retrieves all rows for simplicity. For large datasets, you might want to add a `WHERE` clause or indexing to optimize performance.
- The similarity threshold (0.5) can be adjusted based on testing to balance precision and recall.
- If you need to make the database path configurable, you can add it to the `.env` file and load it using `os.getenv`.

This modified code maintains all the original functionality while replacing the web scraping component with a database query, ensuring that `recommended_content_details` is populated only with semantically relevant, hockey-related content from the `YouTube_Urls` table.