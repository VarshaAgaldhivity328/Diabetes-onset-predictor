import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    
    label_encoders = {}
    print("Preprocessing data...")
    for col in ['gender', 'smoking_history']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model (this may take a moment)...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    
    score = rf_classifier.score(X_test, y_test)
    print(f"Model Accuracy on Test Set: {score:.4f}")
    
    print("Saving model and encoders...")
    joblib.dump(rf_classifier, 'diabetes_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("Optimization finished successfully. Saved 'diabetes_model.pkl' and 'label_encoders.pkl'.")
    print("You can now run 'streamlit run app.py' to launch the web application.")

if __name__ == '__main__':
    main()
