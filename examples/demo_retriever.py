DOCS = {
    "paris": "Paris is the capital of France.",
    "lamp": "The desk lamp uses 40 watts.",
    "bananas": "Bananas are a yellow fruit.",
}


def retrieve(query: str, k: int):
    del query
    if int(k) <= 1:
        return [{"id": "lamp", "text": DOCS["lamp"]}]
    return [
        {"id": "paris", "text": DOCS["paris"]},
        {"id": "bananas", "text": DOCS["bananas"]},
    ][: max(int(k), 1)]
