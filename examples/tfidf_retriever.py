import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS = [
    {
        "id": "catalog_notice.txt",
        "text": "The Apollo mission notebooks were digitized in 2019, and the archive website lists their catalog numbers.",
    },
    {
        "id": "special_collections.txt",
        "text": "University Special Collections houses the handwritten mission logs from the moon program in climate-controlled storage.",
    },
    {
        "id": "battery_lab.txt",
        "text": "The battery lab tests sodium cells under controlled cycling conditions.",
    },
    {
        "id": "orchard.txt",
        "text": "The orchard team records apple harvest dates in a shared ledger.",
    },
    {
        "id": "acoustics.txt",
        "text": "The acoustics laboratory measures violin resonance in a silent chamber.",
    },
    {
        "id": "marine.txt",
        "text": "The marine station keeps salinity records beside the instrument room.",
    },
]

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform([doc["text"] for doc in CORPUS])


def retrieve(query: str, k: int):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = np.argsort(-scores, kind="stable")[:k]
    return [
        {
            "id": CORPUS[i]["id"],
            "text": CORPUS[i]["text"],
            "score": float(scores[i]),
        }
        for i in ranked
    ]
