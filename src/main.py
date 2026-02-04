from src.pipelines import process_review

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
        out = process_review(user_review)
        for k, v in out.items():
            print(f"{k}: {v}\n")
        

        

if __name__ == "__main__":
    main()



