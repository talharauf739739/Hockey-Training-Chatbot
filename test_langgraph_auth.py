import logging
import re
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from OpenAPI_DB import agentic_hockey_chat, is_refusal, is_semantic_refusal, is_in_domain, is_greeting_or_vague, hockey_keywords, sentence_model, hockey_reference_embedding, hockey_technique_embedding, hockey_context_embedding
import httpx
import asyncio
from sentence_transformers import util

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the state for the LangGraph workflow
class ChatState(TypedDict):
    user_active_role: str
    user_team: str
    user_prompt: str
    original_prompt: str
    ai_response: str
    recommended_content_details: List[Dict[str, str]]
    is_valid: bool
    error_message: str
    attempt_count: int
    max_attempts: int

# Node to call agentic_hockey_chat
async def call_hockey_chat(state: ChatState) -> ChatState:
    logging.info(f"Attempt {state['attempt_count'] + 1}/{state['max_attempts']} - Calling agentic_hockey_chat with prompt: {state['user_prompt']}")
    try:
        result = await agentic_hockey_chat(
            state["user_active_role"],
            state["user_team"],
            state["user_prompt"]
        )
        state["ai_response"] = result["ai_response"]
        state["recommended_content_details"] = result["recommended_content_details"]
        state["is_valid"] = True
        state["attempt_count"] += 1
    except Exception as e:
        logging.error(f"Error in agentic_hockey_chat: {str(e)}")
        state["ai_response"] = f"Error processing request: {str(e)}"
        state["recommended_content_details"] = []
        state["is_valid"] = False
        state["error_message"] = str(e)
        state["attempt_count"] += 1
    return state

# Node to validate response accuracy
async def validate_response(state: ChatState) -> ChatState:
    if not state["is_valid"]:
        return state

    ai_response = state["ai_response"]
    recommended_content = state["recommended_content_details"]
    user_lang = "nl" if "nl" in detect(state["original_prompt"]) else "en"

    # Check for greetings or vague prompts
    if is_greeting_or_vague(state["original_prompt"], user_lang):
        expected_response = (
            "Hallo! Waarmee kan ik je helpen met betrekking tot hockey, training of andere onderwerpen?"
            if user_lang == "nl"
            else "Hello! How can I assist you with hockey, training, or other topics?"
        )
        if ai_response != expected_response:
            logging.warning(f"Expected greeting response but got: {ai_response}")
            state["is_valid"] = False
            state["error_message"] = "Invalid greeting response."
            state["ai_response"] = expected_response
            state["recommended_content_details"] = []
            return state

    # Check if response is hockey-related using keywords and semantic similarity
    if not is_in_domain(ai_response):
        response_embedding = sentence_model.encode(ai_response, convert_to_tensor=True)
        primary_similarity = util.cos_sim(response_embedding, hockey_reference_embedding).item()
        technique_similarity = util.cos_sim(response_embedding, hockey_technique_embedding).item()
        context_similarity = util.cos_sim(response_embedding, hockey_context_embedding).item()
        if max(primary_similarity, technique_similarity, context_similarity) < 0.3:
            logging.warning(f"Response is not hockey-related. Similarities: primary={primary_similarity:.3f}, technique={technique_similarity:.3f}, context={context_similarity:.3f}")
            state["is_valid"] = False
            state["error_message"] = "Response is not related to field hockey."
            state["ai_response"] = (
                "Sorry, ik kan alleen helpen met vragen over hockey, zoals training, oefeningen, strategieën, regels en tutorials. Stel me een hockeygerelateerde vraag!"
                if user_lang == "nl"
                else "Sorry, I can only assist with questions about hockey, such as training, drills, strategies, rules, and tutorials. Please ask a hockey-related question!"
            )
            state["recommended_content_details"] = []
            return state

    # Check for refusals
    if is_refusal(ai_response) or is_semantic_refusal(ai_response):
        expected_refusal = (
            "Sorry, ik kan alleen helpen met vragen over hockey, zoals training, oefeningen, strategieën, regels en tutorials. Stel me een hockeygerelateerde vraag!"
            if user_lang == "nl"
            else "Sorry, I can only assist with questions about hockey, such as training, drills, strategies, rules, and tutorials. Please ask a hockey-related question!"
        )
        if ai_response != expected_refusal:
            logging.warning(f"Expected refusal response but got: {ai_response}")
            state["is_valid"] = False
            state["error_message"] = "Invalid refusal response."
            state["ai_response"] = expected_refusal
            state["recommended_content_details"] = []
            return state

    # Validate recommended content
    if not recommended_content:
        logging.warning("No recommendations provided.")
        state["is_valid"] = False
        state["error_message"] = "No valid recommendations provided."
        return state

    query_embedding = sentence_model.encode(state["user_prompt"], convert_to_tensor=True)
    for item in recommended_content:
        if not item.get("title") or not item.get("url"):
            logging.warning("Invalid recommendation format.")
            state["is_valid"] = False
            state["error_message"] = "Invalid recommendation format."
            state["recommended_content_details"] = []
            return state
        if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', item["url"]):
            logging.warning(f"Invalid URL: {item['url']}")
            state["is_valid"] = False
            state["error_message"] = "Invalid URL in recommendations."
            state["recommended_content_details"] = []
            return state
        if not any(keyword in item["title"].lower() for keyword in hockey_keywords):
            logging.warning(f"Recommendation title not hockey-related: {item['title']}")
            state["is_valid"] = False
            state["error_message"] = "Recommendations must be hockey-related."
            state["recommended_content_details"] = []
            return state
        title_embedding = sentence_model.encode(item["title"], convert_to_tensor=True)
        similarity = util.cos_sim(query_embedding, title_embedding).item()
        if similarity < 0.3:
            logging.warning(f"Recommendation not semantically relevant: {item['title']}, similarity={similarity:.3f}")
            state["is_valid"] = False
            state["error_message"] = "Recommendations are not semantically relevant to the prompt."
            state["recommended_content_details"] = []
            return state

    logging.info("Response validated successfully.")
    return state

# Node to authenticate response
async def authenticate_response(state: ChatState) -> ChatState:
    if not state["is_valid"]:
        return state

    ai_response = state["ai_response"].lower()
    sensitive_terms = ["password", "api key", "personal information", "confidential"]
    if any(term in ai_response for term in sensitive_terms):
        logging.warning("Response contains sensitive information.")
        state["is_valid"] = False
        state["error_message"] = "Response contains sensitive information."
        state["ai_response"] = "Sorry, the response contains restricted content."
        state["recommended_content_details"] = []
        return state

    for item in state["recommended_content_details"]:
        if not item["url"].startswith("https://www.youtube.com/"):
            logging.warning(f"Non-YouTube URL detected: {item['url']}")
            state["is_valid"] = False
            state["error_message"] = "Recommendations must come from YouTube."
            state["recommended_content_details"] = []
            return state

    logging.info("Response authenticated successfully.")
    return state

# Node to decide if regeneration is needed and refine prompt
async def regenerate_response(state: ChatState) -> ChatState:
    if not state["is_valid"] and state["attempt_count"] < state["max_attempts"]:
        logging.info(f"Response invalid, attempting regeneration (attempt {state['attempt_count'] + 1}/{state['max_attempts']})")
        # Refine prompt for next attempt
        if state["attempt_count"] == 1:
            state["user_prompt"] = f"{state['original_prompt']} Please provide field hockey-related information, such as training, drills, strategies, or tutorials."
        elif state["attempt_count"] == 2:
            state["user_prompt"] = f"{state['original_prompt']} Focus strictly on field hockey topics like shooting, passing, or goalkeeping."
        return state
    elif not state["is_valid"]:
        logging.info("Max attempts reached, returning last ai_response.")
        # Return the last ai_response as is, without resetting to error message
    else:
        logging.info("Response is valid, no regeneration needed.")
    return state

# Define the LangGraph workflow
workflow = StateGraph(ChatState)

# Add nodes
workflow.add_node("call_hockey_chat", call_hockey_chat)
workflow.add_node("validate_response", validate_response)
workflow.add_node("authenticate_response", authenticate_response)
workflow.add_node("regenerate_response", regenerate_response)

# Define edges
workflow.add_edge("call_hockey_chat", "validate_response")
workflow.add_edge("validate_response", "authenticate_response")
workflow.add_edge("authenticate_response", "regenerate_response")
workflow.add_conditional_edges(
    "regenerate_response",
    lambda state: "call_hockey_chat" if not state["is_valid"] and state["attempt_count"] < state["max_attempts"] else END,
    {
        "call_hockey_chat": "call_hockey_chat",
        END: END
    }
)

# Set entry point
workflow.set_entry_point("call_hockey_chat")

# Compile the graph
graph = workflow.compile()

# Function to run the LangGraph workflow
async def run_langgraph_auth(user_active_role: str, user_team: str, user_prompt: str) -> Dict:
    initial_state = {
        "user_active_role": user_active_role,
        "user_team": user_team,
        "user_prompt": user_prompt,
        "original_prompt": user_prompt,  # Store original prompt for regeneration
        "ai_response": "",
        "recommended_content_details": [],
        "is_valid": False,
        "error_message": "",
        "attempt_count": 0,
        "max_attempts": 3
    }
    result = await graph.ainvoke(initial_state)
    return {
        "ai_response": result["ai_response"],
        "recommended_content_details": result["recommended_content_details"]
    }

if __name__ == "__main__":
    # Example test
    async def test():
        test_cases = [
            {"user_active_role": "le Coach", "user_team": "U8C", "user_prompt": "Suggest me Best Shooting Tips for kids."},
            {"user_active_role": "le Coach", "user_team": "U10C", "user_prompt": "Hey! what's the weather update in Netherlands?"},
            {"user_active_role": "le Coach", "user_team": "U10C", "user_prompt": "Hey!"}
        ]
        for case in test_cases:
            result = await run_langgraph_auth(
                user_active_role=case["user_active_role"],
                user_team=case["user_team"],
                user_prompt=case["user_prompt"]
            )
            print(f"Prompt: {case['user_prompt']}\nResult: {result}\n")

    asyncio.run(test())
    