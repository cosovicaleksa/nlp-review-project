from src.language_prediction.predict import predict_language
from src.star_predicton.predict import predict_star

def main():

    user_review = input('Enter review: ')
    user_review, predicted_language = predict_language(user_review)

    if predicted_language == 'Unsupported':
        print("Unsupported language. Review must be in English, German, French, or Spanish.")
    else:
        # Correct language
        print(predicted_language)

        # Star pred
        predicted_star = predict_star(user_review, predicted_language)
        print(predicted_star)

if __name__ == "__main__":
    main()



