from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.models.train_evaluate import evaluate_model, train_model

def run_grid_search(X_train_vec, y_train, X_test, y_test, models, param_grids, lang_name=""):

    print(f"\n Grid Search for language: {lang_name}\n")

    results = {}

    for (name, model), params in zip(models, param_grids):
        print(f"\nTraining model: {name}\n")

        grid = GridSearchCV(estimator=model, param_grid=params, scoring="accuracy", cv=3, n_jobs=-1, verbose=1)

        grid.fit(X_train_vec, y_train)

        print(f"\nBest params for {name}: {grid.best_params_}")
        print(f"Best CV accuracy: {grid.best_score_:.4f}\n")

        best_model = grid.best_estimator_
        trained_model = train_model(best_model, X_train_vec, y_train)  

        eval_results = evaluate_model(trained_model, X_test, y_test)

        results[name] = {
            "best_params": grid.best_params_,
            "best_score": float(grid.best_score_),
            "test_accuracy": float(eval_results["test_accuracy"]),
            "classification_report": eval_results["class_report"],
            "trained_model": trained_model 
        }


    return results



def run_random_search(X_train_vec, y_train, X_test, y_test, models, param_grids, n_iters, lang_name=""):

    print(f"\n Random Search for language: {lang_name} \n")

    results = {}

    for (name, model), params, n_iter in zip(models, param_grids, n_iters):
        print(f"\nTraining model: {name}\n")

        search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=n_iter, cv=2, scoring="accuracy", n_jobs=-1,verbose=1)

        search.fit(X_train_vec, y_train)

        print(f"\nBest params for {name}: {search.best_params_}")
        print(f"Best CV accuracy: {search.best_score_:.4f}\n")

        best_model = search.best_estimator_
        trained_model = train_model(best_model, X_train_vec, y_train) 
        eval_results = evaluate_model(trained_model, X_test, y_test) 

        results[name] = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "test_accuracy": float(eval_results["test_accuracy"]),
            "classification_report": eval_results["class_report"],
            "trained_model": trained_model 
        }


    return results
