import unittest
from utils.data_processing import ConversationProcessor

class TestDataPipeline(unittest.TestCase):
    def test_preprocess_message(self):
        processor = ConversationProcessor()
        conversation = {
            "conversation_id": "test_conv",
            "messages": [
                {"sender": "agent", "text": "Hello, I have an issue with my order."}
            ]
        }
        processed = processor.preprocess_conversation(conversation)
        self.assertIn("quality_scores", processed)
        self.assertIn("overall_score", processed["quality_scores"])
        self.assertEqual(processed["conversation_id"], "test_conv")
        self.assertTrue(len(processed["messages"]) > 0)
        # Check that tokens are extracted
        self.assertIsInstance(processed["messages"][0]["tokens"], list)
        # Check sentiment keys
        self.assertIn("compound", processed["messages"][0]["sentiment"])

if __name__ == "__main__":
    unittest.main()