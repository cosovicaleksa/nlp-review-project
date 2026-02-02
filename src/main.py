from src.language_prediction.predict import predict_language
from src.star_predicton.predict import predict_star
from src.machine_translation.translation import translate
from src.clustering.predict import predict_cluster
from src.summarization.summarization import summarize
from src.sentiment_analysis.predict import sentiment_analysis

texts = [
    # 1) ENGLISH — Cluster 1 (Books) - Expected stars: (4)
    "This book was a pleasant surprise. The story flows naturally from chapter to chapter, and the characters feel believable rather than exaggerated. I especially liked how the author slowly builds emotional tension without forcing it. The ending stayed with me for a while after I finished reading, which is always a good sign. While it is not a masterpiece or something I would reread every year, it is a short but emotionally rich read that I genuinely enjoyed and would recommend to others who like character-driven stories.",

    # 2) SPANISH — Cluster 9 (Kids’ Toys / Gifts) — Expected stars: (5)
    "Compré este juguete para el cumpleaños de mi sobrina y fue todo un éxito. Desde el primer momento le encantaron los colores y la forma del juguete. El material parece seguro, resistente y de buena calidad, incluso después de varios días de uso intenso. Ha pasado horas jugando sin aburrirse y sin que el juguete se rompa o pierda piezas. Me parece una excelente opción como regalo para niños pequeños y sin duda volvería a comprarlo.",

    # 3) GERMAN — Cluster 25 (Clothing Fit Issues) — Expected stars:  (2)
    "Ich habe diese Jacke in meiner üblichen Größe bestellt und war ehrlich gesagt ziemlich enttäuscht. Laut Größentabelle hätte sie perfekt passen sollen, doch in der Realität waren die Ärmel viel zu kurz und die Schultern saßen extrem eng. Der gesamte Schnitt wirkt unausgewogen und unbequem, besonders beim Bewegen. Zwar fühlt sich der Stoff hochwertig an und macht auf den ersten Blick einen guten Eindruck, aber die schlechte Passform ruiniert das Tragegefühl komplett. Nach nur wenigen Minuten musste ich die Jacke wieder ausziehen, weil sie überall drückte. Für diesen Preis hätte ich deutlich mehr erwartet. Leider geht die Jacke zurück.",

    # 4) FRENCH — Cluster 26 (Smell / Odor Complaints) -  Expected stars:  (1)
    "Le produit avait une odeur chimique extrêmement forte dès l’ouverture du paquet. J’ai d’abord pensé que l’odeur disparaîtrait avec le temps, mais même après l’avoir laissé aérer pendant toute une journée, elle persistait toujours. L’odeur devenait même plus désagréable lorsqu’on s’approchait du produit. Il était totalement impossible de l’utiliser dans cet état, surtout dans un espace fermé. Compte tenu de cette mauvaise expérience, j’ai finalement décidé de demander un remboursement."
]


def main():

    for user_review in texts:
        print("\n================ NEW REVIEW ================\n")

        # Language detection 
        predicted_language = predict_language(user_review)
        print(f"Detected language:\n→ {predicted_language}\n")

        # Star prediction 
        predicted_star = predict_star(user_review, predicted_language)
        print(f"Predicted rating:\n→ {predicted_star} stars\n")

        # Translation (if needed)
        if predicted_language != "English":
            translated_user_review = translate(user_review, predicted_language)
            print(f"Translated review (English):\n→ {translated_user_review}\n")
        else:
            translated_user_review = user_review
            print("Translation:\n→ Not needed (already English)\n")

        # Clustering
        predicted_cluster = predict_cluster(translated_user_review)
        print(f"Predicted cluster:\n→ {predicted_cluster}\n")

        # Abstractive summarization (BART)
        abstractive_summary = summarize(translated_user_review, method="bart")
        if abstractive_summary != translated_user_review:
            print(f"Abstractive summary:\n→ {abstractive_summary}\n")
        else:
            print("Abstractive summary:\n→ Text too short for summarization\n")

        # Extractive summarization (TextRank)
        extractive_summary = summarize(translated_user_review, method="textrank")
        if extractive_summary != translated_user_review:
            print(f"Extractive summary:\n→ {extractive_summary}\n")
        else:
            print("Extractive summary:\n→ Text too short for summarization\n")

        #SENTIMENT
        is_english = (predicted_language == "English")
        is_summarized = (abstractive_summary != translated_user_review)

        # sentiment is computed on the same text the message describes
        sentiment_input = abstractive_summary if is_summarized else translated_user_review
        sentiment = sentiment_analysis(sentiment_input)

        if is_english:
            if is_summarized:
                print("\nSentiment of the summarized English review is: ", sentiment)
            else:
                print("\nSentiment of the original English review is: ", sentiment)
        else:
            if is_summarized:
                print("\nSentiment of the translated and summarized text is: ", sentiment)
            else:
                print("\nSentiment of the translated (not summarized) text is: ", sentiment)


        

if __name__ == "__main__":
    main()



