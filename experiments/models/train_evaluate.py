
from sklearn.metrics import classification_report, accuracy_score

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(trained_model, X_test, y_test):
    preds = trained_model.predict(X_test)

    acc = accuracy_score(y_true = y_test, y_pred = preds)

    class_report = classification_report(y_test, preds, output_dict=True)
    results = {
        'preds':preds,
        'test_accuracy': acc,
        'class_report': class_report
        }

    return results