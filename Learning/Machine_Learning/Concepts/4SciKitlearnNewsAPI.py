import json
import os
from pathlib import Path

import requests



API_KEY = os.environ.get("NEWS_API_KEY")
if not API_KEY:
    raise RuntimeError("NEWS_API_KEY is not set")



url = (

    f"https://newsapi.org/v2/top-headlines?"

    f"country=us&pageSize=20&apiKey={API_KEY}"

)



response = requests.get(url)

data = response.json()



if data["status"] != "ok":

    print(data)

    exit()



news = []



for article in data["articles"]:

    news.append({

        "title": article.get("title"),

        "description": article.get("description"),

        "content": article.get("content"),

        "author": article.get("author"),

        "source": article["source"]["name"],

        "published_at": article.get("publishedAt"),

        "url": article.get("url")

    })



output_file = Path(__file__).resolve().parents[3] / "Data" / "Raw" / "news.json"
with output_file.open("w", encoding="utf-8") as file:

    json.dump(news, file, indent=4, ensure_ascii=False)



print(f"Saved {len(news)} articles to {output_file}")