"""
Train ML models for agent response quality classification.
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from utils.data_processing import ConversationProcessor

class ConversationClassifier:
    def __init__(self):
        self.processor = ConversationProcessor()
        self.model = LogisticRegression(random_state=42, max_iter=1000)

    def prepare_training_data(self, conversations):
        X = []
        y = []
        for conv in conversations:
            # Preprocess conversation messages if not already processed
            if 'quality_scores' not in conv:
                conv = self.processor.preprocess_conversation(conv)
            agent_msgs = [m for m in conv['messages'] if m['sender'] == 'agent']
            for msg in agent_msgs:
                features = self._extract_features(msg)
                X.append(features)
                # Label: 1 if quality_score > 0.6 else 0 (binary classification)
                quality_score = (
                    msg.get('empathy_score', 0) +
                    msg.get('politeness_score', 0) +
                    (msg.get('sentiment', {}).get('compound', 0) + 1) / 2
                ) / 3
                label = 1 if quality_score > 0.6 else 0
                y.append(label)
        return np.array(X), np.array(y)

    def _extract_features(self, message):
        return [
            message.get('sentiment', {}).get('compound', 0),
            message.get('textblob_sentiment', {}).get('polarity', 0),
            message.get('textblob_sentiment', {}).get('subjectivity', 0),
            message.get('empathy_score', 0),
            message.get('politeness_score', 0),
            min(message.get('response_time_ms', 0) / 60000, 1.0),  # capped at 1.0
            len(message.get('tokens', [])) / 50,  # normalized token count
        ]

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        os.makedirs("models/trained", exist_ok=True)
        model_path = "models/trained/logistic_regression_classifier.pkl"
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        return self.model

def train_from_processed_data():
    processed_path = "data/processed/processed_chats.json"
    if not os.path.exists(processed_path):
        print(f"Processed data not found at {processed_path}. Please run data preprocessing first.")
        return
    with open(processed_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    classifier = ConversationClassifier()
    X, y = classifier.prepare_training_data(conversations)
    classifier.train(X, y)

if __name__ == "__main__":
    train_from_processed_data()