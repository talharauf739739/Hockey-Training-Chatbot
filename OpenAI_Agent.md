Sure! Here's a concise 10-line explanation of your code:

1. **Environment & Logging Setup**: Loads environment variables (like API keys) using `dotenv` and sets up logging for debugging and tracking.
2. **Keyword-Based Filtering**: Defines rich lists of hockey-specific and out-of-domain keywords to identify whether the user's query is hockey-related.
3. **Embeddings for Semantic Matching**: Uses `SentenceTransformer` embeddings to semantically detect refusals, hockey relevance, or off-topic prompts.
4. **Language Detection & Translation**: Detects the user’s language and translates responses using `deep_translator` for better multilingual support.
5. **DuckDuckGo Search**: Scrapes top hockey-relevant results (like YouTube, PDFs, Wikipedia) using BeautifulSoup for content recommendation.
6. **Refusal Detection**: Detects AI refusals both through keywords and semantic similarity, to handle and rephrase non-informative answers.
7. **Prompt Construction**: Builds a prompt including current query, past conversation, and relevant context to query OpenRouter’s GPT-4o model.
8. **API Call to OpenRouter**: Sends the formatted prompt to the GPT-4o model using `httpx.AsyncClient` and retrieves the AI-generated response.
9. **Response Filtering & Translation**: Filters out URLs, checks for refusals again, translates response to the detected language, and updates chat history.
10. **Returns Response + Recommendations**: Responds with the AI answer and up to 3 relevant web links, ensuring all content is strictly hockey-related.

