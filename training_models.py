import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



def load_or_train_models():
    if not os.path.exists("weather_data.pkl"):
        df = pd.read_csv("weather_data.csv")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, "weather_data.pkl")
    if not os.path.exists("breast-cancer.pkl"):
        df = pd.read_csv("breast-cancer.csv")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, "breast-cancer.pkl")
    if not os.path.exists("BostonHousing.pkl"):
        df = pd.read_csv("BostonHousing.csv")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, "BostonHousing.pkl")
    if not os.path.exists("cricket_scores.pkl"):
        df = pd.read_csv("cricket_scores.csv")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, "cricket_scores.pkl")
 #   if not os.path.exists("spam.pkl"):
    # Load dataset with correct encoding
        #df = pd.read_csv("Z:\genai\multiprediction_project\spam.csv", encoding="latin1")  

    # Check column names
 #       print(df.head())

    # Assume last column is the target
 #       X = df.iloc[:, :-1]  # Features (all columns except last)
        #y = df.iloc[:, -1]   # Target (last column)

    # Convert categorical text data into numbers
 #       for col in X.select_dtypes(include=['object']).columns:
  #          X[col] = LabelEncoder().fit_transform(X[col])

   #     y = LabelEncoder().fit_transform(y)  # Convert target labels

    # Split data
    #    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
     #   model = RandomForestClassifier(n_estimators=100, random_state=42)
      #  model.fit(X_train, y_train)

    # Save model
       # joblib.dump(model, "spam.pkl")


    
    print("Models Trained!!")

load_or_train_models()
    
    
    