# app.py
#sajjad bhai api key
#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

#my api key
#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

#DB Table's Name: YouTube_Urls

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from OpenAPI_DB import agentic_hockey_chat
#from test_1 import agentic_hockey_chat

app = FastAPI(
   title="Agentic Hockey Chat API",
   description="API for Hockey Q&A and recommendations using an OpenRouter agent.",
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


class BotResponse(BaseModel):
   ai_response: str
   recommended_content_details: list = []  # Changed to list to match agentic_hockey_chat output


@app.get("/ask", response_model=BotResponse)
async def ask_get(query: UserQuery):  # Use UserQuery as a dependency for consistency
   result = await agentic_hockey_chat(query.user_active_role, query.user_team, query.user_prompt)
   return BotResponse(**result)


@app.post("/ask", response_model=BotResponse)
async def ask_post(query: UserQuery):
   result = await agentic_hockey_chat(query.user_active_role, query.user_team, query.user_prompt)
   return BotResponse(**result)

