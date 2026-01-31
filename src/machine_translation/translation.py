from src.config import MACHINE_TRANSLATION_MODELS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


_MODEL_CACHE = {}
def load_model(language: str):
    if language not in _MODEL_CACHE:
                
        model_name = MACHINE_TRANSLATION_MODELS[language]
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _MODEL_CACHE[language] = (tokenizer, model)
    
    return _MODEL_CACHE[language]

def translate(user_review: str, language: str):
    tokenizer, model = load_model(language)

     # Tokenize and translate
    inputs = tokenizer([user_review], return_tensors="pt", padding=True, truncation=True)  # uses PyTorch tensors
    translated_vector = model.generate(**inputs)

    # Decode
    translated_text = tokenizer.decode(translated_vector[0], skip_special_tokens=True)

    return translated_text