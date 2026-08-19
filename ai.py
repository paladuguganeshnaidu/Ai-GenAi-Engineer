import sys
import re
import os
from openai import OpenAI
from ddgs import DDGS

# 1. API Configuration
API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Free AI Web Search Script",
    }
)

# 2. Verified 100% Free Models
FREE_MODELS = {
    "1": "poolside/laguna-xs-2.1:free",
    "2": "cohere/north-mini-code:free",
    "3": "google/gemma-4-26b-a4b-it:free",
    "4": "openrouter/free",
    "5": "openai/gpt-oss-20b:free"
}

def clean_search_query(query: str) -> str:
    """Removes conversational fluff so search engine gives accurate results."""
    clean_q = re.sub(r'^(did you know|who is|tell me about|do you know|can you search for)\s+', '', query, flags=re.IGNORECASE)
    return clean_q.strip(" ?!")

def perform_free_web_search(query: str, max_results: int = 4) -> str:
    search_term = clean_search_query(query)
    print(f"🔎 Searching the web for: '{search_term}'...")
    try:
        results = list(DDGS().text(search_term, max_results=max_results))
        if not results:
            return "No web results found."
        
        search_context = ""
        for i, res in enumerate(results, 1):
            search_context += f"\n[Source {i}]\nTitle: {res.get('title')}\nInfo: {res.get('body')}\nLink: {res.get('href')}\n"
        return search_context
    except Exception as e:
        print(f"⚠️ Web search failed: {e}")
        return ""

def main():
    print("==================================================")
    print("      🆓 FREE AI & INTERNET SEARCH TOOLBOX       ")
    print("==================================================")
    
    print("\nSelect a Free Model:")
    for num, model_name in FREE_MODELS.items():
        print(f" [{num}] {model_name}")

    choice = input("\nEnter model choice (1-5, default is 1): ").strip()
    selected_model = FREE_MODELS.get(choice, FREE_MODELS["1"])
    print(f"✅ Selected Model: {selected_model}")

    user_prompt = input("\n💬 Enter your prompt: ").strip()
    if not user_prompt:
        print("Prompt cannot be empty!")
        return

    enable_search = input("🌐 Enable free web search for this prompt? (y/n): ").strip().lower() == 'y'

    messages = []
    
    # System prompt enforcing clean terminal output (no messy tables)
    system_rules = (
        "You are a helpful assistant responding in a plain terminal.\n"
        "STRICT FORMATTING RULES:\n"
        "1. Do NOT use Markdown tables (they look messy in terminal).\n"
        "2. Use clean bullet points and clear headings with bold text.\n"
        "3. Keep language purely in English."
    )

    if enable_search:
        search_results = perform_free_web_search(user_prompt)
        system_rules += (
            "\n\nUse the following web search data to formulate your summary:\n"
            f"--- WEB SEARCH RESULTS ---\n{search_results}\n---------------------------"
        )

    messages.append({"role": "system", "content": system_rules})
    messages.append({"role": "user", "content": user_prompt})

    print(f"\n⏳ Querying {selected_model}...\n")

    try:
        completion = client.chat.completions.create(
            model=selected_model,
            max_tokens=1000,
            messages=messages
        )

        response_text = completion.choices[0].message.content

        print("=================== RESPONSE ===================")
        print(f"Model Used : {selected_model}")
        print(f"Prompt     : {user_prompt}")
        print("------------------------------------------------")
        print(response_text)
        print("================================================")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()