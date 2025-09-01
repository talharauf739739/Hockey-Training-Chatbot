import logging
import os
import re
from dotenv import load_dotenv
import httpx
from langdetect import detect
from deep_translator import GoogleTranslator
import sqlite3
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
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

# Hockey-specific translation dictionary
hockey_translation_dict = {
    "schiettips": "shooting tips",
    "schieten": "shooting",
    "backhand": "backhand",
    "backhandschoten": "backhand shooting",
    "achterhand": "backhand",
    "veldhockey": "field hockey",
    "strafcorner": "penalty corner",
    "sleepflick": "drag flick",
    "doelman": "goalkeeper",
    "aanvaller": "forward",
    "verdediger": "defender",
    "middenvelder": "midfielder",
    "stickbeheersing": "stick handling",
    "balbeheersing": "ball control",
    "hockeyoefeningen": "hockey drills",
    "oefeningen": "drills",
    "kinderen": "kids",
    "verbeteren": "improve"
}

# Expanded hockey keywords for domain detection
hockey_keywords = [
    "hockey", "field hockey", "veldhockey", "match", "wedstrijd", "game", "spel", "goal", "doelpunt",
    "score", "scoren", "ball", "bal", "stick", "hockeystick", "field", "veld", "turf", "kunstgras",
    "pitch", "speelveld", "corner", "short corner", "long corner", "korte hoek", "lange hoek",
    "penalty", "strafbal", "shootout", "strookschot", "penalty stroke", "strafslag",
    "coach", "trainer", "goalkeeper", "doelman", "keeper", "goalie", "defender", "verdediger",
    "midfielder", "middenvelder", "forward", "aanvaller", "striker", "spits", "captain", "aanvoerder",
    "player", "speler", "team", "ploeg",
    "shooting", "schieten", "schiet", "backhand shooting", "backhandschoten", "passing", "passen",
    "backhand", "achterhand", "forehand", "voorhand", "drag flick", "sleeppush", "push pass",
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
    "job", "baan", "education", "onderwijs",
    "ice hockey", "ijshockey", "slap shot", "wrist shot"
]

# Greetings for detection
greetings = [
    "hey", "hello", "hi", "hiya", "yo", "what's up", "sup", "good morning", "good afternoon",
    "good evening", "good night", "howdy", "greetings", "morning", "evening", "hallo", "hoi",
    "goedemorgen", "goedemiddag", "goedenavond", "goedennacht", "hé", "joe", "moi", "dag",
    "goedendag", "aloha", "ciao", "salut", "hola", "heej"
]

# Common Dutch question starters (not greetings)
dutch_question_starters = [
    "geef me", "kun je", "kunt u", "hoe kan", "wat is", "waarom", "welke", "hoe moet", "wat zijn"
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

# Semantic detection setup with multilingual model
sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

refusal_embedding = sentence_model.encode(
    "Sorry, I can only assist with questions about field hockey, such as training, drills, strategies, rules, and tutorials. Please ask a field hockey-related question!",
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
    "Questions about politics, other sports like football, ice hockey, tennis, weather, movies, music, cooking, or unrelated general topics.",
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

def preprocess_prompt(prompt: str, user_lang: str) -> tuple[str, str]:
    """
    Preprocess prompt and return both translated (English) and original prompt.
    """
    if not prompt or not isinstance(prompt, str):
        return prompt, prompt
    prompt_lower = prompt.lower().strip()
    if user_lang == "nl":
        # Apply hockey-specific translations
        for dutch_term, english_term in hockey_translation_dict.items():
            prompt_lower = re.sub(rf'\b{re.escape(dutch_term)}\b', english_term, prompt_lower)
        try:
            translated = GoogleTranslator(source="nl", target="en").translate(prompt_lower)
            logging.debug(f"Translated Dutch prompt '{prompt_lower}' to English: '{translated}'")
            return translated if translated else prompt_lower, prompt
        except Exception as e:
            logging.error(f"Translation error for prompt '{prompt_lower}': {str(e)}")
            return prompt_lower, prompt
    return prompt_lower, prompt

def is_in_domain(prompt: str) -> bool:
    if not prompt or not isinstance(prompt, str):
        logging.debug("Prompt is empty or not a string.")
        return False
    prompt_lower = prompt.lower().strip()

    has_hockey_keywords = any(
        re.search(rf'\b{re.escape(word)}\b|\b{re.escape(word[:-1])}\w*\b', prompt_lower)
        for word in hockey_keywords
    )
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
            hockey_primary_similarity > 0.3 or 
            hockey_technique_similarity > 0.3 or 
            hockey_context_similarity > 0.3)

def is_greeting_or_vague(prompt: str, user_lang: str) -> bool:
    if not prompt or not isinstance(prompt, str):
        logging.debug("Prompt is empty or not a string.")
        return True
    prompt_lower = prompt.lower().strip()
    is_greeting = any(greeting in prompt_lower for greeting in greetings)
    is_question_starter = any(starter in prompt_lower for starter in dutch_question_starters) if user_lang == "nl" else False
    has_hockey_keywords = any(
        re.search(rf'\b{re.escape(word)}\b|\b{re.escape(word[:-1])}\w*\b', prompt_lower)
        for word in hockey_keywords
    )

    logging.debug(f"Vague check (lang={user_lang}): is_greeting={is_greeting}, "
                  f"is_question_starter={is_question_starter}, has_hockey_keywords={has_hockey_keywords}")

    return is_greeting and not (is_question_starter or has_hockey_keywords)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_youtube_urls_db(english_query: str, dutch_query: str) -> list:
    if not is_in_domain(english_query):
        logging.info("Query is out of domain, skipping database search.")
        return []

    try:
        # Encode both English and Dutch queries
        english_embedding = sentence_model.encode(english_query, convert_to_tensor=False)
        english_embedding = np.array(english_embedding).astype("float32").reshape(1, -1)
        faiss.normalize_L2(english_embedding)

        dutch_embedding = sentence_model.encode(dutch_query, convert_to_tensor=False) if dutch_query else english_embedding
        dutch_embedding = np.array(dutch_embedding).astype("float32").reshape(1, -1)
        faiss.normalize_L2(dutch_embedding)

        # Search with both embeddings, no limit on search scope
        top_k = len(metadata)  # Search all entries
        distances_en, indices_en = index.search(english_embedding, top_k)
        distances_nl, indices_nl = index.search(dutch_embedding, top_k) if dutch_query else (distances_en, indices_en)

        results = []
        seen_urls = set()
        field_hockey_terms = ["field hockey", "veldhockey"]
        ice_hockey_terms = ["ice hockey", "ijshockey", "slap shot", "wrist shot"]

        # Combine results from both searches
        for indices, distances in [(indices_en, distances_en), (indices_nl, distances_nl)]:
            for idx, sim in zip(indices[0], distances[0]):
                if idx < len(metadata) and sim > 0.3:  # Include results above threshold
                    title = metadata[idx]["title"].lower()
                    url = metadata[idx]["url"]
                    logging.debug(f"FAISS match: title='{metadata[idx]['title']}', similarity={sim:.3f}")
                    if (any(term in title for term in field_hockey_terms) or 
                        not any(term in title for term in ice_hockey_terms)) and url not in seen_urls:
                        results.append({
                            "title": metadata[idx]["title"],  # Keep original title (English or Dutch)
                            "url": url,
                            "similarity": float(sim)
                        })
                        seen_urls.add(url)

        # Return only the top 5 results by similarity
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]
        logging.info(f"FAISS search completed with {len(results)} results (limited to top 5).")
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
        {"question": "Hoe train je voor strafcorners?", 
         "answer": "Oefen sleepflicks en ingestudeerde spelsituaties met focus op timing en precisie."},
        {"question": "What are good drills for improving backhand shooting?", 
         "answer": "Use cone shooting drills and practice wrist flicks for power and accuracy."},
        {"question": "Geef me oefeningen voor backhandschoten voor kinderen", 
         "answer": "Gebruik kegeloefeningen en laat kinderen polsbewegingen oefenen voor kracht en precisie."}
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

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if not text or not isinstance(text, str):
        logging.debug("Empty or invalid text for translation, returning empty string.")
        return ""
    if source_lang == target_lang:
        return text
    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        logging.debug(f"Translated text from {source_lang} to {target_lang}: {translated}")
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
        if user_lang not in ["en", "nl"]:
            logging.info(f"Detected language {user_lang} not supported, defaulting to English.")
            user_lang = "en"
    except Exception:
        user_lang = "en"
        logging.debug("Language detection failed, defaulting to English.")

    # Get both translated and original prompts
    processing_prompt, original_prompt = preprocess_prompt(user_prompt, user_lang)
    logging.info(f"Processing prompt after translation: {processing_prompt}")

    if is_greeting_or_vague(user_prompt, user_lang):
        answer = "Hallo! Waarmee kan ik je helpen met betrekking tot hockey, training of andere onderwerpen?" if user_lang == "nl" else "Hello! How can I assist you with hockey, training, or other topics?"
        update_conversation_history(user_active_role, user_team, user_prompt, answer)
        return {"ai_response": answer, "recommended_content_details": []}

    if not is_in_domain(processing_prompt):
        answer = "Sorry, ik kan alleen helpen met vragen over hockey, zoals training, oefeningen, strategieën, regels en tutorials. Stel me een hockeygerelateerde vraag!" if user_lang == "nl" else "Sorry, I can only assist with questions about hockey, such as training, drills, strategies, rules, and tutorials. Please ask a hockey-related question!"
        update_conversation_history(user_active_role, user_team, user_prompt, answer)
        return {"ai_response": answer, "recommended_content_details": []}

    history = get_conversation_history(user_active_role, user_team)
    context = get_relevant_context(processing_prompt)

    system_prompt = (
        "You are an AI Assistant Bot specialized in all things field hockey, including training, drills, strategies, rules, and more. "
        "You communicate with a {user_active_role} from the team {user_team}. "
        "Provide concise, practical, and specific answers tailored to the user's role and team, especially for youth teams like U8C. "
        "Focus on field hockey-related topics such as training, drills, strategies, rules, and tutorials. "
        "Ensure the response is semantically accurate and relevant to the question.\n\n"
        "Recent conversation:\n{history}\n\n"
        "Relevant previous conversations:\n{context}\n\n"
        "Answer the following question in English based on the provided context and your expertise:\n{user_prompt}"
    )

    hockey_prompt_template = system_prompt.format(
        user_active_role=user_active_role,
        user_team=user_team,
        history=history or "No previous conversations.",
        context=context or "No relevant context available.",
        user_prompt=processing_prompt
    )

    payload = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": hockey_prompt_template}
        ],
        "max_tokens": 200,  # Increased for more detailed responses
        "temperature": 0.3,  # Slightly higher for better variety
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
            # Translate answer to the user's language
            answer = translate_text(answer, "en", user_lang)

            logging.info("Performing FAISS search...")
            recommended_content = search_youtube_urls_db(processing_prompt, original_prompt if user_lang == "nl" else "")
            logging.info(f"FAISS search completed with {len(recommended_content)} results.")

            if is_refusal(answer) or is_semantic_refusal(answer):
                logging.warning(f"Response flagged as refusal: {answer}")
                answer = "Sorry, ik kan alleen helpen met vragen over hockey, zoals training, oefeningen, strategieën, regels en tutorials. Stel me een hockeygerelateerde vraag!" if user_lang == "nl" else "Sorry, I can only assist with questions about hockey, such as training, drills, strategies, rules, and tutorials. Please ask a hockey-related question!"
                recommended_content = []

            # Filter out similarity from recommended content
            filtered_recommended_content = [
                {"title": item["title"], "url": item["url"]}
                for item in recommended_content
            ]

            update_conversation_history(user_active_role, user_team, user_prompt, answer)
            return {"ai_response": answer, "recommended_content_details": filtered_recommended_content}

    except httpx.HTTPStatusError as e:
        logging.error(f"OpenRouter API error: Status {e.response.status_code}, Response: {e.response.text}")
        return {"ai_response": f"API error: {e.response.text}", "recommended_content_details": []}
    except Exception as e:
        logging.error(f"Internal error: {str(e)}")
        return {"ai_response": f"Internal error: {str(e)}", "recommended_content_details": []}