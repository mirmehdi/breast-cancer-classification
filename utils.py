import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class Classification: 
    def __init__(self, X, y): 
        self.X = X.copy()
        self.y = y.copy()
        self.models = {
            "SVM": SVC(),
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier()
        }

    # Preprocessing 
    def preprocess(self): 
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42
        )

    # Train and evaluate
    def train_evaluate(self):
        print("=== Model Evaluation ===")
        
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            print(f"\n{name}:")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1 Score:  {f1:.4f}")

    # Cross-validation
    def cross_validate(self, cv=5): 
        print("\n=== Cross-Validation Scores ===")

        best_score = 0.0
        for name, model in self.models.items():
            scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='f1')
            mean_score = scores.mean()
            print(f"{name}: Mean Accuracy = {mean_score:.4f}")
        
            if mean_score > best_score:
                best_score = mean_score
                self.best_model_name = name
                self.best_model = model

        print(f"\n✅ Best model based on cross-validation: {self.best_model_name}")

    # Improving of the model 
    def optimization_best_model(self):
        print(f"\n=== Grid Search for {self.best_model_name} ===")

        if self.best_model_name == "Random Forest":
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5]
            }
        elif self.best_model_name == "SVM":
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        elif self.best_model_name == "KNN":
            param_grid = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        elif self.best_model_name == "Logistic Regression":
            param_grid = {
                'C': [0.1, 1.0, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
        else:
            print("No recognized model to optimize.")
            return

        grid_search = GridSearchCV(
            estimator=self.best_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)
        self.best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Final predictions
        y_pred = self.best_model.predict(self.X_test)

        # Output
        print("\n✅ Best Parameters Found:")
        print(best_params)

        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        print(f"\n🧪 Final Test Accuracy: {acc:.4f}")
        print(f"🧪 Final Test F1 Score: {f1:.4f}")

        return y_pred, best_params
