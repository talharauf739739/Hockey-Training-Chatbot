import logging
import os
import re
import string
from dotenv import load_dotenv
import httpx
from langdetect import detect
from deep_translator import GoogleTranslator
import sqlite3
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
import torch
import pickle
import faiss
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DATABASE_PATH = os.getenv("DATABASE_PATH", "HockeyFood.db")
EMBEDDINGS_PATH = "video_embeddings.npy"
METADATA_PATH = "video_metadata.json"

if not OPENROUTER_API_KEY:
    logging.error("OPENROUTER_API_KEY not set in .env file.")
    raise RuntimeError("OPENROUTER_API_KEY not set in .env file.")
else:
    masked_key = OPENROUTER_API_KEY[:6] + "..." + OPENROUTER_API_KEY[-4:]
    logging.info(f"Loaded OpenRouter API key: {masked_key}")

if not os.path.exists(DATABASE_PATH):
    logging.error(f"Database file not found at {DATABASE_PATH}.")
    raise FileNotFoundError(f"Database file not found at {DATABASE_PATH}.")

# In-memory conversation history
conversation_histories = {}

# Preload embeddings and build FAISS index
index = None
metadata = []
try:
    if not (os.path.exists(EMBEDDINGS_PATH) and os.path.exists(METADATA_PATH)):
        logging.info("Generating embeddings and metadata from database...")
        embeddings = []
        metadata = []
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT title, url, embedding FROM YouTube_Urls")
        for title, url, embedding_blob in cursor.fetchall():
            if title and url and embedding_blob:
                try:
                    embedding = pickle.loads(embedding_blob)
                    if isinstance(embedding, np.ndarray):
                        embeddings.append(embedding)
                        metadata.append({"title": title[:100], "url": url})
                except Exception as e:
                    logging.debug(f"Skipping invalid embedding: {e}")
        conn.close()
        
        if embeddings:
            embeddings_np = np.array(embeddings, dtype=np.float32)
            np.save(EMBEDDINGS_PATH, embeddings_np)
            with open(METADATA_PATH, "w") as f:
                json.dump(metadata, f)
            logging.info(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_PATH} and metadata to {METADATA_PATH}")
        else:
            logging.error("No valid embeddings found in database.")
            raise RuntimeError("No valid embeddings found in database.")

    # Load embeddings and metadata
    embeddings_np = np.load(EMBEDDINGS_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)
    logging.info(f"Built FAISS index with {embeddings_np.shape[0]} embeddings of dimension {dimension}")
except Exception as e:
    logging.error(f"Error initializing FAISS index: {str(e)}")
    raise

# Expanded hockey keywords for domain detection
hockey_keywords = [
    "hockey", "field hockey", "veldhockey", "match", "wedstrijd", "game", "spel", "goal", "doelpunt",
    "score", "scoren", "ball", "bal", "stick", "hockeystick", "field", "veld", "turf", "kunstgras",
    "pitch", "speelveld", "corner", "short corner", "long corner", "korte hoek", "lange hoek",
    "penalty", "strafbal", "shootout", "strookschot", "penalty stroke", "strafslag",
    "coach", "trainer", "goalkeeper", "doelman", "keeper", "goalie", "defender", "verdediger",
    "midfielder", "middenvelder", "forward", "aanvaller", "striker", "spits", "captain", "aanvoerder",
    "player", "speler", "team", "ploeg",
    "shooting", "schieten", "passing", "passen", "backhand", "achterhand", "forehand", "voorhand",
    "wrist shot", "pols schot", "slap shot", "slagschot", "drag flick", "sleeppush", "push pass",
    "pushpass", "hit pass", "slagpass", "aerial pass", "luchtpass", "dribbling", "dribbelen",
    "stick work", "stickwerk", "deflection", "afbuiging", "scoop", "scheppen", "tackle", "tackelen",
    "block tackle", "blok tackle", "jab tackle", "steektackle", "reverse stick", "omgekeerde stick",
    "indian dribble", "indiase dribbel", "3d skills", "3d vaardigheden", "goalkeeping", "doelverdediging",
    "save", "redding", "clearance", "uitverdediging", "flick", "slepen", "lift", "optillen",
    "chip", "chippen", "sweep hit", "veegslag", "tomahawk", "backstick", "reverse hit", "omgekeerde slag",
    "drag", "slepen", "dummy", "schijnbeweging", "feint", "fint", "spin", "draaien",
    "training", "oefening", "exercise", "oefenen", "drill", "oefensessie", "practice", "praktijk",
    "warm-up", "opwarming", "cool-down", "afkoeling", "conditioning", "conditietraining",
    "fitness", "fitheid", "agility", "wendbaarheid", "speed", "snelheid", "endurance", "uithoudingsvermogen",
    "strength", "kracht", "core strength", "kernkracht", "stick handling", "stickbeheersing",
    "ball control", "balbeheersing", "footwork", "voetwerk", "positioning", "positionering",
    "marking", "dekken", "zone defense", "zonedekking", "man-to-man", "man-op-man",
    "attack drill", "aanvalsoefening", "defense drill", "verdedigingsoefening",
    "passing drill", "passoefening", "shooting drill", "schietoefening", "goalkeeper drill",
    "doelmanoefening", "skill development", "vaardigheidsontwikkeling", "technique", "techniek",
    "strategy", "strategie", "tactic", "tactiek", "game plan", "spelplan", "formation", "opstelling",
    "press", "druk zetten", "counterattack", "tegenaanval", "breakaway", "uitbraak",
    "offensive play", "aanvallend spel", "defensive play", "verdedigend spel", "set piece",
    "standaardsituatie", "free hit", "vrije slag", "penalty corner", "strafcorner",
    "tutorial", "handleiding", "tips", "advies", "coaching", "coachen", "learn", "leren",
    "education", "opleiding", "skills training", "vaardigheidstraining", "workshop", "werkplaats",
    "session", "sessie", "clinic", "kliniek", "instruction", "instructie", "guide", "gids",
    "shin guard", "scheenbeschermer", "mouthguard", "mondbeschermer", "gloves", "handschoenen",
    "grips", "grepen", "turf shoes", "kunstschoenen", "hockey shoes", "hockeyschoenen",
    "goalpost", "doelpaal", "net", "netwerk", "training cone", "trainingskegel",
    "rebound board", "reboundbord", "practice net", "oefennet",
    "warmup", "opwarmen", "stretching", "rekken", "injury prevention", "blessurepreventie",
    "teamwork", "samenwerking", "communication", "communicatie", "leadership", "leiderschap",
    "motivation", "motivatie", "mental preparation", "mentale voorbereiding", "focus", "concentratie",
    "hockey camp", "hockeykamp", "tournament", "toernooi", "league", "liga", "championship",
    "kampioenschap"
]

# Out-of-domain keywords
out_of_domain_keywords = [
    "politics", "politiek", "government", "regering", "election", "verkiezing", "policy", "beleid",
    "football", "voetbal", "soccer", "basketball", "basketbal", "tennis", "cricket", "rugby",
    "volleyball", "volleybal", "baseball", "honkbal", "golf", "swimming", "zwemmen",
    "athletics", "atletiek", "cycling", "wielrennen", "boxing", "boksen", "martial arts",
    "vechtsport", "gymnastics", "gymnastiek", "weather", "weer", "temperature", "temperatuur",
    "forecast", "voorspelling", "rain", "regen", "snow", "sneeuw", "storm", "wind", "sun",
    "zon", "cloud", "wolk", "humidity", "vochtigheid", "climate", "klimaat", "pollution",
    "vervuiling", "movie", "film", "television", "televisie", "music", "muziek", "concert",
    "celebrity", "beroemdheid", "news", "nieuws", "gossip", "roddel", "streaming", "streamen",
    "video game", "videospel", "gaming", "gamen", "cooking", "koken", "recipe", "recept",
    "fashion", "mode", "shopping", "winkelen", "travel", "reizen", "vacation", "vakantie",
    "car", "auto", "finance", "financiën", "stock market", "aandelenmarkt", "business", "zaken",
    "job", "baan", "education", "onderwijs"
]

# Greetings for detection
greetings = [
    "hey", "hello", "hi", "hiya", "yo", "what's up", "sup", "good morning", "good afternoon",
    "good evening", "good night", "howdy", "greetings", "morning", "evening", "hallo", "hoi",
    "goedemorgen", "goedemiddag", "goedenavond", "goedennacht", "hé", "joe", "moi", "dag",
    "goedendag", "aloha", "ciao", "salut", "hola", "heej"
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
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

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
    if not text or not isinstance(text, str):
        logging.debug("Empty or invalid text for refusal check.")
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in refusal_keywords)

def is_semantic_refusal(text: str) -> bool:
    if not text or not isinstance(text, str):
        logging.debug("Empty or invalid text for semantic refusal check.")
        return False
    embedding = sentence_model.encode(text, convert_to_tensor=True)
    similarity = util.cos_sim(embedding, refusal_embedding).item()
    logging.debug(f"Semantic refusal similarity: {similarity:.3f}")
    return similarity > 0.7

def is_in_domain(prompt: str) -> bool:
    if not prompt or not isinstance(prompt, str):
        logging.debug("Prompt is empty or not a string.")
        return False
    prompt_lower = prompt.lower().strip()

    has_hockey_keywords = any(word in prompt_lower for word in hockey_keywords)
    has_out_of_domain_keywords = any(word in prompt_lower for word in out_of_domain_keywords)

    prompt_embedding = sentence_model.encode(prompt_lower, convert_to_tensor=True)
    hockey_primary_similarity = util.cos_sim(prompt_embedding, hockey_reference_embedding).item()
    hockey_technique_similarity = util.cos_sim(prompt_embedding, hockey_technique_embedding).item()
    hockey_context_similarity = util.cos_sim(prompt_embedding, hockey_context_embedding).item()

    logging.debug(f"Domain check: has_hockey_keywords={has_hockey_keywords}, "
                  f"has_out_of_domain_keywords={has_out_of_domain_keywords}, "
                  f"primary_sim={hockey_primary_similarity:.3f}, "
                  f"technique_sim={hockey_technique_similarity:.3f}, "
                  f"context_sim={hockey_context_similarity:.3f}")

    if has_out_of_domain_keywords:
        logging.info("Prompt contains out-of-domain keywords, marked as out of domain.")
        return False

    return (has_hockey_keywords or 
            hockey_primary_similarity > 0.4 or 
            hockey_technique_similarity > 0.4 or 
            hockey_context_similarity > 0.4)

def is_greeting_or_vague(prompt: str) -> bool:
    if not prompt or not isinstance(prompt, str):
        logging.debug("Prompt is empty or not a string.")
        return True
    prompt_lower = prompt.lower().strip()
    return any(greeting in prompt_lower for greeting in greetings) or len(prompt_lower.split()) < 3

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_youtube_urls_db(query: str) -> list:
    if not is_in_domain(query):
        logging.info("Query is out of domain, skipping database search.")
        return []

    try:
        query_embedding = sentence_model.encode(query, convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        top_k = 5
        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        for idx, sim in zip(indices[0], distances[0]):
            if idx < len(metadata) and sim > 0.4:  # Apply threshold
                results.append({
                    "title": metadata[idx]["title"],
                    "url": metadata[idx]["url"],
                    "similarity": float(sim)
                })

        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]
        logging.info(f"FAISS search completed with {len(results)} results.")
        return results
    except Exception as e:
        logging.error(f"FAISS search error: {e}")
        return []

def get_conversation_history(user_role: str, user_team: str) -> str:
    session_key = f"{user_role}|{user_team}"
    history = conversation_histories.get(session_key, [])
    formatted_history = "\n".join([f"Gebruiker: {q}\nCoach: {a}" for q, a in history[-3:]])
    logging.debug(f"Conversation history for {session_key}: {formatted_history}")
    return formatted_history

def update_conversation_history(user_role: str, user_team: str, question: str, answer: str):
    session_key = f"{user_role}|{user_team}"
    history = conversation_histories.get(session_key, [])
    history.append((question, answer))
    conversation_histories[session_key] = history[-3:]
    logging.debug(f"Updated conversation history for {session_key} with question: {question}")

def get_relevant_context(question: str) -> str:
    sample_context = [
        {"question": "What are good drills for improving stick handling?", 
         "answer": "Try cone dribbling and figure-eight patterns to enhance stick control."},
        {"question": "How to train for penalty corners?", 
         "answer": "Practice drag flicks and set plays with a focus on timing and accuracy."}
    ]
    question_lower = question.lower() if isinstance(question, str) else ""
    relevant = [
        f"Vraag: {entry['question']}\nAntwoord: {entry['answer']}"
        for entry in sample_context
        if any(kw in question_lower for kw in hockey_keywords) and 
           any(kw in entry['question'].lower() for kw in hockey_keywords)
    ]
    context = "\n\n".join(relevant[:2])
    logging.debug(f"Relevant context for question '{question}': {context}")
    return context

def translate_text(text: str, target_lang: str) -> str:
    if not text or not isinstance(text, str):
        logging.debug("Empty or invalid text for translation, returning empty string.")
        return ""
    if target_lang == "en":
        return text
    try:
        translated = GoogleTranslator(source="en", target=target_lang).translate(text)
        logging.debug(f"Translated text to {target_lang}: {translated}")
        return translated
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def agentic_hockey_chat(user_active_role: str, user_team: str, user_prompt: str) -> dict:
    logging.info(f"Processing question: {user_prompt}, role: {user_active_role}, team: {user_team}")

    # Sanitize user prompt
    if not user_prompt or not isinstance(user_prompt, str):
        logging.error("Invalid or empty user_prompt.")
        return {"ai_response": "Vraag mag niet leeg zijn.", "recommended_content_details": []}
    user_prompt = re.sub(r'\s+', ' ', user_prompt.strip())

    try:
        user_lang = detect(user_prompt)
    except Exception:
        user_lang = "en"
        logging.debug("Language detection failed, defaulting to English.")

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
        "You are an AI Assistant Bot specialized in all things field hockey, including training, drills, strategies, rules, and more. "
        "You communicate in English with a {user_active_role} from the team {user_team}. "
        "Provide concise, practical, and specific answers tailored to the user's role and team, especially for youth teams like U19. "
        "Avoid refusal responses and focus solely on hockey-related topics such as training, drills, strategies, rules, and tutorials.\n\n"
        "Recent conversation:\n{history}\n\n"
        "Relevant previous conversations:\n{context}\n\n"
        "Answer the following question in English based on the provided context and your expertise:\n{user_prompt}"
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
            logging.debug(f"Raw API response: {data}")

            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if not answer:
                logging.error("No answer received from OpenRouter API.")
                return {"ai_response": "No answer received from the API.", "recommended_content_details": []}

            answer = re.sub(r'https?://\S+', '', answer).strip()
            answer = translate_text(answer, user_lang)

            logging.info("Performing FAISS search...")
            recommended_content = search_youtube_urls_db(user_prompt)
            logging.info(f"FAISS search completed with {len(recommended_content)} results.")

            if is_refusal(answer) or is_semantic_refusal(answer):
                logging.warning(f"Response flagged as refusal: {answer}")
                answer = "Sorry, I can only assist with hockey-related questions. Please ask a hockey-related question!" if user_lang == "en" else "Sorry, ik kan alleen helpen met vragen over hockey, zoals training, oefeningen, strategieën, regels en tutorials. Stel me een hockeygerelateerde vraag!"
                recommended_content = []

            update_conversation_history(user_active_role, user_team, user_prompt, answer)
            return {"ai_response": answer, "recommended_content_details": recommended_content}

    except httpx.HTTPStatusError as e:
        logging.error(f"OpenRouter API error: Status {e.response.status_code}, Response: {e.response.text}")
        return {"ai_response": f"API error: {e.response.text}", "recommended_content_details": []}
    except Exception as e:
        logging.error(f"Internal error: {str(e)}")
        return {"ai_response": f"Internal error: {str(e)}", "recommended_content_details": []}