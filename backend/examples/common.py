import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(
    model=os.getenv("LLM_MODEL"),
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    model_provider="openai",
)