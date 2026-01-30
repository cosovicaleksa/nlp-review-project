from src.language_prediction.predict import predict_language

def main():

    user_review = input('Enter review: ')
    user_review, predicted_label = predict_language(user_review)

    if predicted_label == 'Unsupported':
        print("Unsupported language. Review must be in English, German, French, or Spanish.")
    else:
        print(predicted_label)

if __name__ == "__main__":
    main()



