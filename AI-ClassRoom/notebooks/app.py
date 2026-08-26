import os
from dotenv import load_dotenv

load_dotenv()


def get_api_key(provider):
    env_names = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Gemini": "GEMINI_API_KEY",
    }

    api_key = os.getenv(env_names[provider])

    if not api_key:
        raise ValueError(f"{env_names[provider]} is missing from the .env file.")

    return api_key


# ...existing code...

def communicate(provider, api_key, message):
    if provider == "OpenAI":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
        )
        return response.choices[0].message.content

    if provider == "Anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=500,
            messages=[{"role": "user", "content": message}],
        )
        return response.content[0].text

    if provider == "Gemini":
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=message,
        )
        return response.text

    raise ValueError("Unsupported API provider")


providers = ["OpenAI", "Anthropic", "Gemini"]

print("Choose the API provider:")
for index, provider in enumerate(providers, start=1):
    print(f"{index}. {provider}")

choice = int(input("Enter your choice: "))
provider = providers[choice - 1]
api_key = get_api_key(provider)

print(f"\nUsing {provider}. Type 'exit' to stop.")

while True:
    user_message = input("\nYou: ")

    if user_message.lower() == "exit":
        print("Conversation ended.")
        break

    try:
        answer = communicate(provider, api_key, user_message)
        print(f"\nAssistant: {answer}")
    except Exception as error:
        print(f"\nError: {error}")