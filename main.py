from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles # Removed unused import
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn

from bert_chatbot.core.chatbot import SemanticChatBot
# from bert_chatbot.cli import print_welcome_message # For welcome message - REMOVED

# --- Configuration ---
DATABASE_PATH = "DATA_CHAT_2.xlsx" # <--- ИЗМЕНЕНО: Укажите здесь имя файла, который работает в CLI
# Consider making these configurable (e.g., via environment variables or a config file)
SIMILARITY_THRESHOLD = 0.75
CACHE_FILE = "vector_cache.pkl"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# --- Application Setup ---
app = FastAPI(
    title="Bert Chatbot API",
    description="API for interacting with the Bert Semantic Chatbot.",
    version="0.1.0"
)

# --- Global Variables & Initialization ---
# Base path for resolving relative paths
BASE_DIR = Path(__file__).resolve().parent

# Mount static files (CSS, JS) if you have any in a 'static' directory
# app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Initialize the chatbot
# Ensure the database path is correct and accessible
actual_db_path: Path = BASE_DIR / DATABASE_PATH # Initialize before try block
try:
    if not actual_db_path.exists():
        # Attempt to find it relative to a common project structure if not in root
        # This is a guess; adjust if your structure is different
        project_root_db_path = BASE_DIR.parent / DATABASE_PATH # e.g. if main.py is in an 'app' folder
        if project_root_db_path.exists():
            actual_db_path = project_root_db_path
        else:
            # As a last resort, check relative to the current working directory
            # This is less reliable but might catch cases where the script is run from the project root
            cwd_db_path = Path.cwd() / DATABASE_PATH
            if cwd_db_path.exists():
                actual_db_path = cwd_db_path
            else:
                print(f"ERROR: Database file '{DATABASE_PATH}' not found at '{BASE_DIR / DATABASE_PATH}', nor '{project_root_db_path}', nor '{cwd_db_path}'. Please check the path.")
                # You might want to raise an error here or handle it more gracefully
                # For now, we'll let the chatbot initialization fail if the path is truly wrong.
                pass # Let SemanticChatBot handle the FileNotFoundError

    print(f"Attempting to load database from: {actual_db_path}")
    chatbot_instance = SemanticChatBot(
        database_path=str(actual_db_path),
        similarity_threshold=SIMILARITY_THRESHOLD,
        cache_file=str(BASE_DIR / CACHE_FILE), # Cache file in the same directory as main.py
        model_name=MODEL_NAME,
    )
    print("Chatbot initialized successfully.")
    if chatbot_instance and hasattr(chatbot_instance, 'db_vocabulary'):
        print(f"[FastAPI] DB Vocabulary size: {len(chatbot_instance.db_vocabulary)}")
        if chatbot_instance.db_vocabulary:
            sample_vocab = list(chatbot_instance.db_vocabulary)[:10]
            print(f"[FastAPI] DB Vocabulary sample: {sample_vocab}")
        else:
            print("[FastAPI] DB Vocabulary is EMPTY!")
    else:
        print("[FastAPI] Chatbot instance or db_vocabulary not available for logging.")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Database file not found at {actual_db_path}. Chatbot functionality will be severely limited or non-functional.")
    # Fallback or error state for chatbot_instance if needed
    chatbot_instance = None # Or a mock/dummy chatbot
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize SemanticChatBot: {e}")
    import traceback
    traceback.print_exc()
    chatbot_instance = None


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    """Serves the main chat HTML page."""
    # print_welcome_message() # Optional: if you want it in server logs too - REMOVED
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": []})

@app.post("/chat", response_class=HTMLResponse)
async def handle_chat_query(request: Request, user_query: str = Form(...)):
    """
    Handles a user's query, gets a response from the chatbot,
    and returns an HTML snippet to update the chat interface.
    """
    if chatbot_instance is None:
        response_data = {
            "answer": "Chatbot is not available due to an initialization error. Please check server logs.",
            "source": "",
            "matched_question": "",
            "score": 0.0,
            "above_threshold": False,
            "match_type": "error_no_chatbot",
        }
    else:
        try:
            response_data = chatbot_instance.find_answer(user_query)
        except Exception as e:
            print(f"Error during find_answer: {e}")
            import traceback
            traceback.print_exc()
            response_data = {
                "answer": f"An error occurred while processing your request: {e}",
                "source": "",
                "matched_question": "",
                "score": 0.0,
                "above_threshold": False,
                "match_type": "error_processing_query",
            }

    # Prepare context for the HTML snippet
    # The snippet will display the user's query and the bot's answer.
    context = {
        "request": request,
        "user_query": user_query,
        "bot_answer": response_data.get("answer", "No answer found."),
        "source": response_data.get("source", ""),
        "matched_question": response_data.get("matched_question", ""),
        "score": response_data.get("score", 0.0),
        "match_type": response_data.get("match_type", "N/A"),
        "relevant_questions_list": response_data.get("relevant_questions_list", [])
    }
    
    # Log for debugging
    # print(f"User Query: {user_query}")
    # print(f"Bot Response Data: {response_data}")
    # print(f"Context for template: {context}")

    return templates.TemplateResponse("chat_response_snippet.html", context)


# --- Main Execution (for direct run) ---
if __name__ == "__main__":
    # This allows running the app with `python main.py`
    # Uvicorn is a lightning-fast ASGI server.
    print("Starting FastAPI server with Uvicorn...")
    # print_welcome_message() # Print welcome message when server starts - REMOVED
    
    # Ensure the host and port are suitable for your environment
    # "0.0.0.0" makes it accessible on your network, not just localhost
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 