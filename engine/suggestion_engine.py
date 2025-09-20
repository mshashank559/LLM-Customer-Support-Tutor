"""
Suggestion engine for real-time AI coaching in customer support chats.
Tracks conversation state and generates coaching suggestions using an LLM.
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time

class ConversationState:
    """
    Tracks conversation messages and context.
    """
    def __init__(self, conversation_id):
        self.conversation_id = conversation_id
        self.messages = []  # list of dicts: {"sender": "agent"/"customer", "text": str, "timestamp": float}

    def add_message(self, sender, text):
        self.messages.append({
            "sender": sender,
            "text": text,
            "timestamp": time.time()
        })

    def get_context(self, max_tokens=512):
        """
        Returns the conversation history as a single string, truncated to max_tokens.
        """
        convo_text = ""
        for msg in self.messages[-20:]:  # limit to last 20 messages for context
            prefix = "Agent: " if msg["sender"] == "agent" else "Customer: "
            convo_text += prefix + msg["text"] + "\n"
        # Truncate to max_tokens tokens approximately (simple truncation)
        tokens = convo_text.split()
        if len(tokens) > max_tokens:
            tokens = tokens[-max_tokens:]
        return " ".join(tokens)

class SuggestionEngine:
    """
    Generates coaching suggestions using a pretrained LLM.
    """
    def __init__(self, model_name="google/flan-t5-base", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def generate_suggestion(self, conversation_context, category):
        """
        Generate a coaching suggestion for a given category based on conversation context.

        Categories:
            - tone_adjustment
            - empathy
            - technical_accuracy
            - policy_reminder
        """
        prompt = self._build_prompt(conversation_context, category)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=100,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        suggestion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return suggestion.strip()

    def _build_prompt(self, conversation_context, category):
        """
        Build a prompt for the LLM based on the conversation and suggestion category.
        """
        category_prompts = {
            "tone_adjustment": "Suggest how the agent can improve the tone of their messages to be more positive and professional.",
            "empathy": "Suggest ways the agent can show more empathy and understanding towards the customer.",
            "technical_accuracy": "Suggest corrections or improvements to the technical accuracy of the agent's responses.",
            "policy_reminder": "Remind the agent about relevant company policies or compliance requirements."
        }
        base_prompt = (
            "You are an AI assistant helping a customer support agent improve their chat responses.\n"
            "Here is the recent conversation:\n"
            f"{conversation_context}\n\n"
            f"Based on the above, {category_prompts.get(category, '')}\n"
            "Provide a concise suggestion."
        )
        return base_prompt