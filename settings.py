from dotenv import load_dotenv
import os

load_dotenv(".config.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
