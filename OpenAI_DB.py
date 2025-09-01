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
import torch


# Configure logging
logging.basicConfig(level=logging.INFO)


# Load environment variables
load_dotenv()
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


# Hockey keywords for domain detection
hockey_keywords = [
   "hockey", "training", "exercise", "drill", "coach", "plan", "learn", "education", "practice", "skills",
   "shooting", "passing", "goalkeeper", "match", "ball", "stick", "field", "goal", "strategy", "tactic",
   "oefening", "wedstrijd", "stick", "bal", "keeper", "veld", "doelpunt", "oefeningen", "trainen",
   "schieten", "passen", "doelman", "doel", "strategie", "tactiek", "tutorial", "tips"
]


# Out-of-domain keywords (politics, other sports, weather, etc.)
out_of_domain_keywords = [
   "politics", "politiek", "government", "regering", "election", "verkiezing", "policy", "beleid",
   "football", "voetbal", "basketball", "basketbal", "tennis", "cricket", "rugby", "volleyball",
   "weather", "weer", "temperature", "temperatuur", "forecast", "voorspelling", "rain", "regen",
   "snow", "sneeuw", "storm", "storm"
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
sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
refusal_embedding = sentence_model.encode("Sorry, I cannot help with that", convert_to_tensor=True)
hockey_reference_embedding = sentence_model.encode(
   "Questions about hockey training, drills, strategies, rules, or tutorials", convert_to_tensor=True
)
out_of_domain_reference_embedding = sentence_model.encode(
   "Questions about politics, other sports, weather, or unrelated topics", convert_to_tensor=True
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
  
   # Semantic analysis
   prompt_embedding = sentence_model.encode(prompt, convert_to_tensor=True)
   hockey_similarity = util.cos_sim(prompt_embedding, hockey_reference_embedding).item()
   out_of_domain_similarity = util.cos_sim(prompt_embedding, out_of_domain_reference_embedding).item()
  
   # Combine keyword and semantic checks
   is_hockey_related = has_hockey_keywords or hockey_similarity > 0.6
   is_not_out_of_domain = not has_out_of_domain_keywords and out_of_domain_similarity < 0.5
  
   return is_hockey_related and is_not_out_of_domain


# Sample context data
sample_context = [
   {"user_role": "player", "user_team": "team_a", "question": "Hoe verbeter ik mijn slagtechniek?", "answer": "Focus op je grip en houding. Oefen de basisslag met een lage zwaai en volg door."},
   {"user_role": "coach", "user_team": "team_b", "question": "Wat is een goede warming-up voor een wedstrijd?", "answer": "Begin met 5 minuten joggen, gevolgd door dynamische stretches en korte sprints."},
   {"user_role": "player", "user_team": "team_a", "question": "Hoe verdedig ik tegen een snelle aanvker?", "answer": "Blijf laag, gebruik je stick om de bal te blokkeren en anticipeer op hun bewegingen."}
]


def is_greeting_or_vague(prompt: str) -> bool:
   prompt_lower = prompt.lower().strip()
   prompt_clean = prompt_lower.translate(str.maketrans('', '', string.punctuation)).strip()
   return prompt_clean in greetings or len(prompt_clean.split()) < 2


def translate_text(text: str, target_lang: str) -> str:
   try:
       detected_lang = detect(text)
       if detected_lang != target_lang:
           translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
           return translated if translated else text
       return text
   except Exception as e:
       logging.warning(f"Translation failed: {str(e)}")
       return text


def query_youtube_urls_db(query: str) -> list:
   if not is_in_domain(query):
       return []
  
   try:
       # Connect to the SQLite database
       conn = sqlite3.connect('HockeyFood.db')
       cursor = conn.cursor()
      
       # Retrieve all rows from YouTube_Urls
       cursor.execute("SELECT title, url, metatags FROM YouTube_Urls")
       rows = cursor.fetchall()
      
       # Close database connection
       conn.close()
      
       # Compute query embedding
       query_embedding = sentence_model.encode(query, convert_to_tensor=True)
      
       results = []
       allowed_domains = ["youtube.com", "wikipedia.org", ".pdf", "arxiv.org", "zenodo.org"]
      
       for title, url, metatags in rows:
           if not title or not url:
               continue
              
           # Combine title and metatags for semantic analysis
           content_text = f"{title} {metatags or ''}".strip()
           if not content_text:
               continue
              
           # Check for refusals
           if is_refusal(title) or is_refusal(url) or is_semantic_refusal(title):
               continue
              
           # Check if URL is from an allowed domain and contains hockey keywords
           if not any(domain in url.lower() for domain in allowed_domains):
               continue
              
           # Compute semantic similarity
           content_embedding = sentence_model.encode(content_text, convert_to_tensor=True)
           similarity = util.cos_sim(query_embedding, content_embedding).item()
          
           # Apply hockey keyword and semantic domain checks
           has_hockey_keywords = any(kw in title.lower() or (metatags and kw in metatags.lower()) for kw in hockey_keywords)
           content_hockey_similarity = util.cos_sim(content_embedding, hockey_reference_embedding).item()
          
           if has_hockey_keywords or content_hockey_similarity > 0.6:
               results.append({
                   "title": title[:100],
                   "url": url,
                   "similarity": similarity
               })
      
       # Sort by similarity and take top 5
       results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]
      
       # Remove similarity key from final output
       final_results = [{"title": r["title"], "url": r["url"]} for r in results]
      
       return final_results
      
   except sqlite3.Error as e:
       logging.error(f"Database error: {str(e)}")
       return []
   except Exception as e:
       logging.error(f"Error querying database: {str(e)}")
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
       user_lang = "nl"


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


   if user_lang == "nl":
       system_prompt = (
           "Je bent een AI Assistent Bot, gespecialiseerd in alles wat met hockey te maken heeft, inclusief training, oefeningen, strategieën, regels en meer. "
           "Je communiceert in het Nederlands met een {user_active_role} van het team {user_team}.\n\n"
           "Hier is een recent gesprek:\n{history}\n\n"
           "Hier zijn enkele eerdere hockeygesprekken die relevant kunnen zijn:\n{context}\n\n"
           "Beantwoord de volgende vraag in het Nederlands op basis van de verstrekte context en je expertise. "
           "Focus uitsluitend op hockeygerelateerde onderwerpen zoals training, oefeningen, strategieën, regels en tutorials.\n{user_prompt}"
       )
   else:
       system_prompt = (
           "You are an AI Assistant Bot specialized in all things hockey, including training, drills, strategies, rules, and more. "
           "You communicate in English with a {user_active_role} from the team {user_team}.\n\n"
           "Here is a recent conversation:\n{history}\n\n"
           "Here are some previous hockey-related conversations that may be relevant:\n{context}\n\n"
           "Answer the following question in English based on the provided context and your expertise. "
           "Focus solely on hockey-related topics such as training, drills, strategies, rules, and tutorials.\n{user_prompt}"
       )


   hockey_prompt_template = system_prompt.format(
       user_active_role=user_active_role,
       user_team=user_team,
       history=history or "No previous conversations." if user_lang != "nl" else "Geen eerdere gesprekken.",
       context=context or "No relevant context available." if user_lang != "nl" else "Geen context beschikbaar.",
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
       async with httpx.AsyncClient(timeout=20) as client:
           response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
           response.raise_for_status()
           data = response.json()
           answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
           if not answer:
               logging.error("No answer received from OpenRouter API.")
               return {"ai_response": "Geen antwoord ontvangen van de API." if user_lang == "nl" else "No answer received from the API.", "recommended_content_details": []}


           answer = re.sub(r'https?://\S+', '', answer).strip()
           answer = translate_text(answer, user_lang)


           recommended_content = query_youtube_urls_db(user_prompt)


           if is_refusal(answer) or is_semantic_refusal(answer):
               answer = "Sorry, ik kan alleen helpen met vragen over hockey. Stel me een hockeygerelateerde vraag!" if user_lang == "nl" else "Sorry, I can only assist with hockey-related questions. Please ask a hockey-related question!"
               recommended_content = []


           update_conversation_history(user_active_role, user_team, user_prompt, answer)


           return {"ai_response": answer, "recommended_content_details": recommended_content}


   except httpx.HTTPStatusError as e:
       logging.error(f"OpenRouter API error: {e.response.text}")
       return {"ai_response": f"API-fout: {e.response.text}" if user_lang == "nl" else f"API error: {e.response.text}", "recommended_content_details": []}
   except Exception as e:
       logging.error(f"Internal error: {str(e)}")
       return {"ai_response": f"Interne fout: {str(e)}" if user_lang == "nl" else f"Internal error: {str(e)}", "recommended_content_details": []}
