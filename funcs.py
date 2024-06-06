import os
from dotenv import load_dotenv, find_dotenv
import openai


def get_openai_key():
    _ = load_dotenv(find_dotenv())
    return os.environ['OPENAI_API_KEY']


openai.api_key = get_openai_key()
print(openai.api_key)