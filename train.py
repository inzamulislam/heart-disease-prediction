import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# data loading
df = pd.read_csv('heart.csv')

# feature and target separating
X = df.drop('target', axis=1)
y = df['target']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# model save
if not os.path.exists('model'):
    os.makedirs('model')

model_path = 'model/heart_model.joblib'
joblib.dump(model, model_path)

print(f"Model trained and saved at {model_path}")