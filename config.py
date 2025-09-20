# config.py
"""
Configuration settings for the Customer Support Chat Tutor project.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

LLM_MODEL = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLASSIFICATION_MODEL = "logistic_regression"

QUALITY_WEIGHTS = {
    "greeting": 0.15,
    "problem_identification": 0.25,
    "solution_delivery": 0.35,
    "closing": 0.15,
    "empathy": 0.10
}

MAX_RESPONSE_TIME_MS = 2000  # 2 seconds max for suggestions
CONVERSATION_HISTORY_LENGTH = 10