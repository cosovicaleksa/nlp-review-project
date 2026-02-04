import ollama

client = ollama.Client()
MODEL = "mistral"


def mistral_star_predictor(text: str) -> int:
    prompt = f"""
    You are a review rating classifier.
    Output only a number from 1 to 5.

    Review: "{text}"
    """
    response = client.generate(model=MODEL, prompt=prompt)
    return int(response["response"].strip())


def mistral_translate(text: str) -> str:
    prompt = f"""
    Translate the review to English if needed.
    Output only the translated text.

    Review: "{text}"
    """
    response = client.generate(model=MODEL, prompt=prompt)
    return response["response"].strip()


def mistral_cluster(text: str) -> str:
    prompt = f"""
    Choose EXACTLY ONE cluster from the predefined list.
    Do not explain.

    Review: "{text}"
    """
    response = client.generate(model=MODEL, prompt=prompt)
    return response["response"].strip()


def mistral_sentiment_analysis(text: str) -> str:
    prompt = f"""
    Classify sentiment: positive, neutral, or negative.
    Output only the label.

    Review: "{text}"
    """
    response = client.generate(model=MODEL, prompt=prompt)
    return response["response"].strip()


def mistral_extractive_summary(text: str) -> str:
    prompt = f"""
    Produce a short extractive summary.
    Use only words from the original text.

    Review: "{text}"
    """
    response = client.generate(model=MODEL, prompt=prompt)
    return response["response"].strip()


def mistral_abstractive_summary(text: str) -> str:
    prompt = f"""
    Produce a short abstractive summary.

    Review: "{text}"
    """
    response = client.generate(model=MODEL, prompt=prompt)
    return response["response"].strip()
