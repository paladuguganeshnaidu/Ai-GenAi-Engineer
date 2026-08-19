import os

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

API_KEY = os.environ.get("DETECT_LANGUAGE_API_KEY")
if not API_KEY:
    raise RuntimeError("DETECT_LANGUAGE_API_KEY is not set")

texts = [
    "Hello, how are you?",
    "Good morning everyone",
    "Bonjour tout le monde",
    "Merci beaucoup",
    "Hola amigo",
    "Buenos dias",
    "Guten Morgen",
    "Wie geht es dir",
    "Ciao amico",
    "Buongiorno",
    "Namaste kaise ho",
    "Dhanyavaad",
    "Konnichiwa",
    "Arigato",
    "Ni hao",
    "Xie xie",
    "Ela unnav?",
    "Thinnava?"
]

labels = []

url = "https://ws.detectlanguage.com/0.2/detect"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

for text in texts:
    response = requests.post(
        url,
        headers=headers,
        data={"q": text}
    )

    data = response.json()

    language = data["data"]["detections"][0]["language"]

    labels.append(language)

    print(f"{text} --> {language}")

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,4))

X = vectorizer.fit_transform(texts)

model = MultinomialNB()

model.fit(X, labels)

print("\nModel trained successfully!")

while True:

    sentence = input("\nEnter a sentence (or 'exit'): ")

    if sentence.lower() == "exit":
        break

    X_test = vectorizer.transform([sentence])

    prediction = model.predict(X_test)[0]

    response = requests.post(
        url,
        headers=headers,
        data={"q": sentence}
    )

    api_prediction = response.json()["data"]["detections"][0]["language"]

    print("Scikit-learn Prediction :", prediction)
    print("API Prediction          :", api_prediction)