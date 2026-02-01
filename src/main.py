from src.language_prediction.predict import predict_language
from src.star_predicton.predict import predict_star
from src.machine_translation.translation import translate
from src.clustering.predict import predict_cluster

def main():

    user_review = input('Enter review: ')
    user_review, predicted_language = predict_language(user_review)

    # Correct language
    print(f"\n Detected language:\n→ {predicted_language}\n")

    # Star pred
    predicted_star = predict_star(user_review, predicted_language)
    print(f"Predicted rating:\n→ {predicted_star} stars\n")

    if predicted_language != 'English':
        user_review = translate(user_review, predicted_language)
        print(f"Translated review (English):\n→ {user_review}\n")
    
    predicted_cluster = predict_cluster(user_review)
    print(f"Predicted cluster:\n→ {predicted_cluster}\n")

        

if __name__ == "__main__":
    main()



