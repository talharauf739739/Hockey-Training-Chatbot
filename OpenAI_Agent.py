import logging
import os
import re
import string
import urllib.parse
from dotenv import load_dotenv
import httpx
from langdetect import detect
from deep_translator import GoogleTranslator
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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
"""hockey_keywords = [
    "hockey", "training", "exercise", "drill", "coach", "plan", "learn", "education", "practice", "skills",
    "shooting", "passing", "goalkeeper", "match", "ball", "stick", "field", "goal", "strategy", "tactic",
    "oefening", "wedstrijd", "stick", "bal", "keeper", "veld", "doelpunt", "oefeningen", "trainen",
    "schieten", "passen", "doelman", "doel", "strategie", "tactiek", "tutorial", "tips", "penalty", "shootout",
    "backhand", "forehand", "wrist shot", "slap shot", "drag flick", "push pass", "hit pass", "aerial pass",
    "dribbling", "stick work", "deflection", "scoop", "tackle", "block tackle", "jab tackle", "reverse stick",
    "indian dribble", "3d skills", "goalkeeping", "save", "clearance", "corner", "short corner", "long corner"
]"""
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
    "standaardsituatie", "free hit", "vrije slag", "penalty corner", "strafcorner",
    
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
"""
# Out-of-domain keywords
out_of_domain_keywords = [
    "politics", "politiek", "government", "regering", "election", "verkiezing", "policy", "beleid",
    "football", "voetbal", "basketball", "basketbal", "tennis", "cricket", "rugby", "volleyball",
    "weather", "weer", "temperature", "temperatuur", "forecast", "voorspelling", "rain", "regen", "snow",
    "sneeuw", "storm"
]

# Greetings for detection
greetings = [
    "hey", "hello", "hi", "good morning", "good evening",
    "hallo", "hoi", "goedemorgen", "goedenavond"
]

# Refusal detection setup
refusal_keywords = [
    "i can't help", "not available", "cannot provide", "inappropriate", "refuse", "not allowed",
    "no access", "ai cannot respond", "ask something else", "outside my domain", "as an ai language model"
]
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lighter model for performance
refusal_embedding = sentence_model.encode("Sorry, I cannot help with that", convert_to_tensor=True)
hockey_reference_embedding = sentence_model.encode(
    "Questions about hockey training, drills, strategies, rules, or tutorials", convert_to_tensor=True
)
hockey_technique_embedding = sentence_model.encode(
    "Field hockey techniques including backhand shooting, forehand passing, drag flick, dribbling, and tackling",
    convert_to_tensor=True
)
out_of_domain_embedding = sentence_model.encode(
    "Questions about politics, other sports, weather, or unrelated topics", convert_to_tensor=True
)
"""
# Out-of-domain keywords (expanded to cover more non-hockey topics)
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

# Greetings for detection (expanded with informal and regional variations)
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

# Refusal detection keywords (expanded to cover more AI refusal patterns)
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
    
    # Lowered threshold to 0.5 for broader inclusivity
    return has_hockey_keywords or hockey_primary_similarity > 0.5 or hockey_technique_similarity > 0.5

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def duckduckgo_search(query: str) -> list:
    if not is_in_domain(query):
        logging.info("Query is out of domain, skipping search.")
        return []
    
    url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        allowed_domains = ["youtube.com", "wikipedia.org", ".pdf", "arxiv.org", "zenodo.org"]
        for a in soup.select('.result__a')[:10]:
            raw_link = a.get('href')
            title = a.get_text()
            if not raw_link or not title:
                continue
            if raw_link.startswith("//duckduckgo.com/l/?uddg="):
                cleaned_url = urllib.parse.unquote(raw_link.split("uddg=")[1].split("&")[0])
            else:
                cleaned_url = raw_link
            if cleaned_url.startswith("http"):
                url_to_use = cleaned_url
            else:
                url_to_use = "https:" + cleaned_url if cleaned_url.startswith("//") else cleaned_url

            if is_refusal(title) or is_refusal(url_to_use) or is_semantic_refusal(title):
                continue

            if any(domain in url_to_use for domain in allowed_domains) and any(kw in title.lower() or kw in url_to_use.lower() for kw in hockey_keywords):
                results.append({"title": title[:100], "url": url_to_use})
            if len(results) >= 3:
                break
        logging.info(f"Search completed with {len(results)} results.")
        return results
    except Exception as e:
        logging.error(f"DuckDuckGo search error: {str(e)}")
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
    question_lower = question.lower()
    relevant = [
        f"Vraag: {entry['question']}\nAntwoord: {entry['answer']}"
        for entry in sample_context
        if any(kw in question_lower for kw in hockey_keywords) and any(kw in entry['question'].lower() for kw in hockey_keywords)
    ]
    return "\n\n".join(relevant[:2])

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

            logging.info("Performing DuckDuckGo search with 30-second timeout...")
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(duckduckgo_search, user_prompt)
                try:
                    recommended_content = future.result(timeout=30)
                except TimeoutError:
                    recommended_content = future.result() if future.done() else []
            logging.info(f"DuckDuckGo search completed with {len(recommended_content)} results.")

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