import unittest
from advanced_prompting import AdvancedPromptEngineer, PromptEngineeringConfig

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        # Initialize the prompting system without API keys as we won't call OpenAI
        self.config = PromptEngineeringConfig()
        self.engineer = AdvancedPromptEngineer(self.config)
        
        # Override retrieve_external_info with a mock that we can control
        self.mock_external_result = ""
        self.mock_external_exception = False
        
        def mock_retrieve(task):
            if self.mock_external_exception:
                raise Exception("Network Error")
            return self.mock_external_result
            
        self.engineer.retrieve_external_info = mock_retrieve

    def test_internal_and_external(self):
        self.engineer.knowledge_base["task"] = "internal context"
        self.mock_external_result = "external context"
        result = self.engineer.retrieve_information("task")
        self.assertIn("internal context", result)
        self.assertIn("external context", result)

    def test_internal_only(self):
        self.engineer.knowledge_base["task"] = "internal context"
        self.mock_external_result = ""
        result = self.engineer.retrieve_information("task")
        self.assertEqual("internal context", result)

    def test_external_only(self):
        # No KB entry
        self.mock_external_result = "external context"
        result = self.engineer.retrieve_information("task")
        self.assertEqual("external context", result)

    def test_external_failure(self):
        self.engineer.knowledge_base["task"] = "internal context"
        self.mock_external_exception = True
        result = self.engineer.retrieve_information("task")
        self.assertEqual("internal context", result)

    def test_no_sources(self):
        self.mock_external_result = ""
        result = self.engineer.retrieve_information("task")
        self.assertEqual("", result)

    def test_retrieval_propagates_into_generate_initial_prompt(self):
        """Verify that retrieved information propagates into generate_initial_prompt."""
        from advanced_prompting import OutputType
        self.engineer.knowledge_base["sample task"] = "KNOWLEDGE_BASE_DATA"
        self.mock_external_result = "EXTERNAL_API_DATA"
        
        retrieved = self.engineer.retrieve_information("sample task")
        _, prompt = self.engineer.generate_initial_prompt(
            task="sample task",
            retrieved_info=retrieved,
            step_budget=5,
            complexity=1,
            output_type=OutputType(output_type="text", file_extension=".txt"),
        )

        self.assertIn("KNOWLEDGE_BASE_DATA", prompt)
        self.assertIn("EXTERNAL_API_DATA", prompt)

if __name__ == "__main__":
    unittest.main()
