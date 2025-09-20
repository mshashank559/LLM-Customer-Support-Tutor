import unittest
from utils.data_processing import ConversationProcessor

class TestQualityScoring(unittest.TestCase):
    def test_quality_scores(self):
        processor = ConversationProcessor()
        conversation = {
            "conversation_id": "test",
            "messages": [
                {"sender": "agent", "text": "Hello! How can I help you?"},
                {"sender": "customer", "text": "I have a problem with my internet."},
                {"sender": "agent", "text": "I will help you fix that."},
                {"sender": "agent", "text": "Thank you, have a nice day!"}
            ]
        }
        processed = processor.preprocess_conversation(conversation)
        scores = processed["quality_scores"]
        self.assertEqual(scores["greeting"], 1.0)
        self.assertEqual(scores["problem_identification"], 1.0)
        self.assertEqual(scores["solution_delivery"], 1.0)
        self.assertEqual(scores["closing"], 1.0)
        self.assertTrue(0.0 <= scores["empathy"] <= 1.0)
        self.assertTrue(0.0 <= scores["overall_score"] <= 1.0)

if __name__ == "__main__":
    unittest.main()