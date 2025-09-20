"""
Synthetic data generation for customer support conversations.
"""

import json
import random
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

class SyntheticDataGenerator:
    def __init__(self):
        self.model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.topics = [
            "internet connection issues", "billing problems", "account login",
            "product not working", "shipping delay", "refund request",
            "technical support", "order cancellation", "password reset",
            "service outage", "payment failure", "damaged product"
        ]
        self.customer_phrases = [
            "I'm having trouble with {topic}.",
            "My {topic} isn't working properly.",
            "Can you help me with {topic}?",
            "I need assistance with {topic}.",
            "There's an issue with my {topic}."
        ]
        self.agent_phrases = [
            "Hello! How can I assist you today?",
            "Thank you for contacting support. How can I help?",
            "I'm sorry you're experiencing issues. Let me help you.",
            "I understand your frustration. Let's resolve this together.",
            "I appreciate your patience while we work on this."
        ]

    def generate_conversation(self, conversation_id, max_turns=8):
        topic = random.choice(self.topics)
        conversation = []
        agent_msg = random.choice(self.agent_phrases)
        conversation.append({"sender": "agent", "text": agent_msg})
        customer_msg = random.choice(self.customer_phrases).format(topic=topic)
        conversation.append({"sender": "customer", "text": customer_msg})
        current_sender = "agent"
        for _ in range(2, max_turns):
            if current_sender == "agent":
                prompt = self._create_agent_prompt(conversation, topic)
                response = self._generate_response(prompt)
                conversation.append({"sender": "agent", "text": response})
                current_sender = "customer"
            else:
                prompt = self._create_customer_prompt(conversation, topic)
                response = self._generate_response(prompt)
                conversation.append({"sender": "customer", "text": response})
                current_sender = "agent"
        if conversation[-1]["sender"] == "customer":
            prompt = self._create_agent_prompt(conversation, topic)
            response = self._generate_response(prompt)
            conversation.append({"sender": "agent", "text": response})
        return {
            "conversation_id": conversation_id,
            "topic": topic,
            "messages": conversation
        }

    def _create_agent_prompt(self, conversation, topic):
        prompt = f"Generate a helpful customer support agent response about {topic}.\n\n"
        for msg in conversation:
            role = "Customer" if msg["sender"] == "customer" else "Agent"
            prompt += f"{role}: {msg['text']}\n"
        prompt += "Agent:"
        return prompt

    def _create_customer_prompt(self, conversation, topic):
        prompt = f"Generate a customer message about {topic} issue.\n\n"
        for msg in conversation:
            role = "Customer" if msg["sender"] == "customer" else "Agent"
            prompt += f"{role}: {msg['text']}\n"
        prompt += "Customer:"
        return prompt

    def _generate_response(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

def generate_dataset(num_conversations=100, output_path="data/synthetic/synthetic_chats.json"):
    generator = SyntheticDataGenerator()
    dataset = []
    for i in range(num_conversations):
        conv = generator.generate_conversation(f"conv_{i+1}")
        dataset.append(conv)
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_conversations} conversations")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {output_path}")
    return dataset

if __name__ == "__main__":
    generate_dataset()