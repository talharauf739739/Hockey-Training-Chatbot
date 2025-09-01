from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from test_langgraph_auth import run_langgraph_auth
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="Agentic Hockey Chat API",
    description="API for Hockey Q&A and recommendations using an OpenRouter agent with LangGraph authentication and regeneration for robust responses.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    user_active_role: str
    user_team: str
    user_prompt: str

    class Config:
        schema_extra = {
            "example": {
                "user_active_role": "le Coach",
                "user_team": "U8C",
                "user_prompt": "Suggest me Best Shooting Tips for kids."
            }
        }

class BotResponse(BaseModel):
    ai_response: str
    recommended_content_details: List[Dict[str, str]] = []

    class Config:
        schema_extra = {
            "example": {
                "ai_response": "For kids, focus on basic shooting techniques like the push shot. Ensure proper grip and stance, and practice aiming at low corners of the goal.",
                "recommended_content_details": [
                    {"title": "Field Hockey Shooting Basics for Kids", "url": "https://www.youtube.com/watch?v=abc123"},
                    {"title": "Youth Hockey Drills", "url": "https://www.youtube.com/watch?v=xyz789"}
                ]
            }
        }

@app.post("/ask", response_model=BotResponse)
async def ask_post(query: UserQuery):
    # Input validation
    if not query.user_prompt.strip():
        logging.warning("Empty prompt received.")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if not query.user_active_role.strip() or not query.user_team.strip():
        logging.warning("Invalid role or team received.")
        raise HTTPException(status_code=400, detail="User role and team must be non-empty.")

    try:
        result = await run_langgraph_auth(
            user_active_role=query.user_active_role,
            user_team=query.user_team,
            user_prompt=query.user_prompt
        )
        if "only assist with questions about hockey" in result["ai_response"]:
            logging.info(f"Refusal response for prompt: {query.user_prompt}")
        elif "How can I assist you with hockey" in result["ai_response"]:
            logging.info(f"Greeting response for prompt: {query.user_prompt}")
        else:
            logging.info(f"Successfully processed hockey-related prompt: {query.user_prompt}")
        return BotResponse(**result)
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}