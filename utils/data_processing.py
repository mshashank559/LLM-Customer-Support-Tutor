"""
Data processing utilities for customer support conversations.
Includes preprocessing, NLP feature extraction, and quality scoring.
"""

import spacy
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import json
import os

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

class ConversationProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sia = SentimentIntensityAnalyzer()
        self.empathy_keywords = {
            'sorry', 'apologize', 'understand', 'frustrating', 'help', 
            'assist', 'resolve', 'appreciate', 'thank', 'welcome'
        }
        self.politeness_markers = {
            'please', 'thank you', 'thanks', 'would you', 'could you',
            'may I', 'appreciate', 'kindly', 'welcome'
        }

    def preprocess_conversation(self, conversation):
        messages = conversation.get('messages', [])
        processed_messages = []
        for i, msg in enumerate(messages):
            text = msg['text']
            sender = msg['sender']
            doc = self.nlp(text)
            sentiment = self.sia.polarity_scores(text)
            textblob_sentiment = TextBlob(text).sentiment
            empathy_score = self._calculate_empathy_score(text)
            politeness_score = self._calculate_politeness_score(text)
            response_time = self._estimate_response_time(i, messages)
            processed_msg = {
                'text': text,
                'sender': sender,
                'tokens': [token.text for token in doc],
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'sentiment': sentiment,
                'textblob_sentiment': {
                    'polarity': textblob_sentiment.polarity,
                    'subjectivity': textblob_sentiment.subjectivity
                },
                'empathy_score': empathy_score,
                'politeness_score': politeness_score,
                'response_time_ms': response_time
            }
            processed_messages.append(processed_msg)
        quality_scores = self._calculate_quality_scores(processed_messages)
        return {
            'conversation_id': conversation.get('conversation_id', 'unknown'),
            'messages': processed_messages,
            'quality_scores': quality_scores
        }

    def _calculate_empathy_score(self, text):
        text_lower = text.lower()
        empathy_count = sum(1 for word in self.empathy_keywords if word in text_lower)
        return min(empathy_count / 3, 1.0)

    def _calculate_politeness_score(self, text):
        text_lower = text.lower()
        politeness_count = sum(1 for word in self.politeness_markers if word in text_lower)
        return min(politeness_count / 2, 1.0)

    def _estimate_response_time(self, index, messages):
        # Placeholder: real timestamps needed for accurate timing
        if index == 0:
            return 0.0
        return 15000.0 if index % 2 == 0 else 10000.0

    def _calculate_quality_scores(self, messages):
        from config import QUALITY_WEIGHTS
        agent_messages = [m for m in messages if m['sender'] == 'agent']
        if not agent_messages:
            return {key: 0.0 for key in QUALITY_WEIGHTS.keys()}
        greeting_score = self._score_greeting(messages)
        problem_score = self._score_problem_identification(messages)
        solution_score = self._score_solution_delivery(messages)
        closing_score = self._score_closing(messages)
        empathy_score = sum(m['empathy_score'] for m in agent_messages) / len(agent_messages)
        scores = {
            'greeting': greeting_score,
            'problem_identification': problem_score,
            'solution_delivery': solution_score,
            'closing': closing_score,
            'empathy': empathy_score
        }
        overall_score = sum(scores[k] * QUALITY_WEIGHTS[k] for k in QUALITY_WEIGHTS)
        scores['overall_score'] = overall_score
        return scores

    def _score_greeting(self, messages):
        greeting_phrases = {'hello', 'hi', 'good morning', 'good afternoon', 'greetings'}
        if messages and messages[0]['sender'] == 'agent':
            text = messages[0]['text'].lower()
            return 1.0 if any(phrase in text for phrase in greeting_phrases) else 0.0
        return 0.0

    def _score_problem_identification(self, messages):
        problem_phrases = {'issue', 'problem', 'trouble', 'error', 'not working', 'help with'}
        customer_msgs = [m for m in messages if m['sender'] == 'customer']
        if customer_msgs:
            text = ' '.join(m['text'].lower() for m in customer_msgs)
            return 1.0 if any(phrase in text for phrase in problem_phrases) else 0.0
        return 0.0

    def _score_solution_delivery(self, messages):
        solution_phrases = {'fix', 'resolve', 'solution', 'answer', 'help you', 'assist with'}
        agent_msgs = [m for m in messages if m['sender'] == 'agent']
        if agent_msgs:
            text = ' '.join(m['text'].lower() for m in agent_msgs)
            return 1.0 if any(phrase in text for phrase in solution_phrases) else 0.0
        return 0.0

    def _score_closing(self, messages):
        closing_phrases = {'thank you', 'thanks', 'goodbye', 'have a nice day', 'welcome'}
        if messages and messages[-1]['sender'] == 'agent':
            text = messages[-1]['text'].lower()
            return 1.0 if any(phrase in text for phrase in closing_phrases) else 0.0
        return 0.0

def run_data_pipeline():
    synthetic_path = os.path.join("data", "synthetic", "synthetic_chats.json")
    processed_path = os.path.join("data", "processed", "processed_chats.json")
    if not os.path.exists(synthetic_path):
        print(f"Synthetic data not found at {synthetic_path}. Please generate it first.")
        return
    with open(synthetic_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    processor = ConversationProcessor()
    processed = [processor.preprocess_conversation(conv) for conv in conversations]
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)
    print(f"Processed data saved to {processed_path}")

if __name__ == "__main__":
    run_data_pipeline()