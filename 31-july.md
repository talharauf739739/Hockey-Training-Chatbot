Here are 10 lines that describe the purpose and functionality of the code you provided:

1. This code defines an intelligent field hockey assistant that responds to user queries using OpenRouter's GPT-4o model.
2. It loads environment variables, such as API keys and database paths, using `dotenv`.
3. It preloads sentence embeddings from a local SQLite database and builds a FAISS index for fast similarity search.
4. Dutch-language prompts are translated into English using `deep-translator`, with special handling for field hockey terms.
5. The assistant filters prompts using keyword matching and semantic similarity to ensure they're relevant to field hockey.
6. It refuses to answer out-of-domain questions like politics, movies, or other sports, using rule-based and semantic checks.
7. The assistant retrieves similar questions from a predefined context and conversation history for better response accuracy.
8. It sends a structured prompt to the OpenRouter API, customizing the system prompt based on user role and team.
9. Responses are translated back into the user's language (Dutch or English) and cleaned of any unnecessary URLs.
10. Optionally, it fetches YouTube video recommendations using FAISS based on the semantic similarity of the user's query.
