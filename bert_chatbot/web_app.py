from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path

# Import the chatbot
from bert_chatbot.core.chatbot import SemanticChatBot # Assuming chatbot.py is in core

# Assuming your project structure has bert-chatbot as a root or a main package folder
# and templates/static are at the project root, not inside bert_chatbot module.
# Adjust paths if your structure is different.
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI()

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    print(f"Warning: Static directory {STATIC_DIR} not found. CSS/JS may not load.")

# Setup templates
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
else:
    templates = None
    print(f"Warning: Templates directory {TEMPLATES_DIR} not found. HTML pages may not load.")

# Initialize chatbot globally (or use a dependency injection system for FastAPI)
# This path needs to be configurable or passed in, hardcoding for now.
# Ensure this Excel file exists or the bot will fail to initialize.
DATABASE_FILE_PATH = PROJECT_ROOT / "test_data_production.xlsx" 

try:
    print(f"Loading chatbot with database: {DATABASE_FILE_PATH}")
    if not DATABASE_FILE_PATH.exists():
        print(f"ERROR: Database file not found at {DATABASE_FILE_PATH}. Chatbot will not work.")
        chatbot_instance = None
    else:
        chatbot_instance = SemanticChatBot(database_path=str(DATABASE_FILE_PATH))
        print("Chatbot initialized successfully.")
except Exception as e:
    print(f"ERROR: Failed to initialize SemanticChatBot: {e}")
    chatbot_instance = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if templates:
        return templates.TemplateResponse("index.html", {"request": request, "chat_history": []})
    return HTMLResponse("<html><body><h1>Templates directory not found.</h1></body></html>")

@app.post("/chat")
async def handle_chat(request: Request):
    form_data = await request.form()
    user_message_value = form_data.get("message")

    if user_message_value is None:
        user_message = "" # Default to empty string if no message is provided
    elif isinstance(user_message_value, str):
        user_message = user_message_value
    else:
        # If it's not a string and not None (e.g., could be UploadFile),
        # we'll try to represent it as a string. For a chat app, this path is less expected
        # for typical text input, but handles the type checker's concern.
        # A real app might read content if it's an UploadFile, but here we just stringify.
        print(f"Warning: Received non-string message of type {type(user_message_value)}. Converting to string.")
        user_message = str(user_message_value) 

    bot_response = "Sorry, the chatbot is not available at the moment."
    if chatbot_instance:
        try:
            chat_result = chatbot_instance.find_answer(user_message)
            bot_response = chat_result.get("answer", "No answer found.")
            # You can also include similarity, matched_question etc. if needed for debugging on client
            # For example: return { "user_message": user_message, "bot_response": bot_response, "debug_info": chat_result}
        except Exception as e:
            print(f"Error during find_answer: {e}")
            bot_response = "An error occurred while processing your message."
    else:
        print("Chatbot instance is not available for POST /chat")

    # This would typically re-render the page with the new chat history
    # For simplicity now, just returning a JSON or simple HTML snippet
    # In a full app, you might return JSON and update via JavaScript,
    # or redirect/re-render the template.
    return {"user_message": user_message, "bot_response": bot_response}

def start_web_server():
    print(f"Attempting to serve from: {PROJECT_ROOT}")
    print(f"Templates directory configured: {TEMPLATES_DIR}")
    print(f"Static directory configured: {STATIC_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # This allows running the web app directly, e.g., python bert_chatbot/web_app.py
    # You might want to integrate this into your main CLI (e.g., a 'serve' command)
    start_web_server() 