from src.language_prediction.predict import predict_language
from src.star_predicton.predict import predict_star
from src.machine_translation.translation import translate
from src.clustering.predict import predict_cluster
from src.summarization.summarization import summarize


def main():
    print("\n================ NEW REVIEW ================\n")

    user_review = input("Enter review: ")

    # Language detection
    user_review, predicted_language = predict_language(user_review)
    print(f"Detected language:\n→ {predicted_language}\n")

    # Star prediction
    predicted_star = predict_star(user_review, predicted_language)
    print(f"Predicted rating:\n→ {predicted_star} stars\n")

    # Translation (if needed)
    if predicted_language != "English":
        user_review = translate(user_review, predicted_language)
        print(f"Translated review (English):\n→ {user_review}\n")
    else:
        print("Translation:\n→ Not needed (already English)\n")

    # Clustering
    predicted_cluster = predict_cluster(user_review)
    print(f"Predicted cluster:\n→ {predicted_cluster}\n")

    # Abstractive summarization (BART)
    abstractive_summary = summarize(user_review, method="bart")
    if abstractive_summary != user_review:
        print(f"Abstractive summary:\n→ {abstractive_summary}\n")
    else:
        print("Abstractive summary:\n→ Text too short for summarization\n")

    # Extractive summarization (TextRank)
    extractive_summary = summarize(user_review, method="textrank")
    if extractive_summary != user_review:
        print(f"Extractive summary:\n→ {extractive_summary}\n")
    else:
        print("Extractive summary:\n→ Text too short for summarization\n")

        

if __name__ == "__main__":
    main()



