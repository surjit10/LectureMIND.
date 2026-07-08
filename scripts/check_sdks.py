import sys
import google.generativeai as genai
import anthropic
import openai

def check_gemini():
    # genai.configure(api_key="fake")
    print("Gemini has list_models?", hasattr(genai, 'list_models'))

def check_anthropic():
    print("Anthropic client has models?", hasattr(anthropic.Anthropic, 'models'))

def check_openai():
    print("OpenAI client has models?", hasattr(openai.OpenAI, 'models'))

check_gemini()
check_anthropic()
check_openai()
