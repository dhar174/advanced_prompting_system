"""
Test module for conversation_manager.py import integrity.
Issue #27: Verifies that conversation_manager.py is syntactically valid
and importable, with correct symbol ordering and no namespace conflicts.
"""

import importlib
import py_compile
import sys
import os
import unittest


class TestConversationManagerImportIntegrity(unittest.TestCase):
    """Verify conversation_manager.py compiles, imports, and has correct symbols."""

    MODULE_PATH = os.path.join(
        os.path.dirname(__file__), "conversation_manager.py"
    )

    def test_syntax_compilation(self):
        """conversation_manager.py must be syntactically valid Python."""
        try:
            py_compile.compile(self.MODULE_PATH, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"conversation_manager.py has syntax errors: {e}")

    def test_import_succeeds(self):
        """conversation_manager module must import without errors."""
        try:
            module = sys.modules.get("conversation_manager") or importlib.import_module("conversation_manager")
            self.assertIsNotNone(module)
        except Exception as e:
            self.fail(f"conversation_manager failed to import: {e}")

    def test_output_type_exists(self):
        """OutputType class must be available after import."""
        import conversation_manager
        self.assertTrue(hasattr(conversation_manager, "OutputType"))
        self.assertIsNotNone(conversation_manager.OutputType)

    def test_default_output_type_exists_and_is_instance(self):
        """DEFAULT_OUTPUT_TYPE must be an OutputType instance defined after the class."""
        import conversation_manager
        self.assertTrue(hasattr(conversation_manager, "DEFAULT_OUTPUT_TYPE"))
        self.assertIsNotNone(conversation_manager.DEFAULT_OUTPUT_TYPE)
        self.assertIsInstance(
            conversation_manager.DEFAULT_OUTPUT_TYPE,
            conversation_manager.OutputType,
        )

    def test_pydantic_validation_error_not_shadowed(self):
        """The custom validation error must not shadow pydantic.ValidationError."""
        from pydantic import ValidationError as PydanticValidationError
        import conversation_manager

        # The local class should have a distinct name
        self.assertTrue(
            hasattr(conversation_manager, "ConversationValidationError"),
            "Expected ConversationValidationError class to exist",
        )

        # Verify there is no class named exactly 'ValidationError' in the module
        # that is NOT the pydantic one
        module_attrs = dir(conversation_manager)
        if "ValidationError" in module_attrs:
            # It should be the pydantic one (from the import), not a local subclass
            module_ve = getattr(conversation_manager, "ValidationError")
            self.assertIs(
                module_ve,
                PydanticValidationError,
                "Module-level ValidationError should be pydantic's, not a local subclass",
            )

    def test_key_public_apis_available(self):
        """Key public functions and classes must be importable."""
        import conversation_manager

        expected_attrs = [
            "OutputType",
            "DEFAULT_OUTPUT_TYPE",
            "ErrorResult",
            "ConversationManagerError",
            "ConversationValidationError",
            "OutputTypeError",
            "LLMCallError",
            "safe_get_message_field",
            "safe_get_feedback_field",
            "validate_message_structure",
        ]
        for attr in expected_attrs:
            self.assertTrue(
                hasattr(conversation_manager, attr),
                f"Expected conversation_manager.{attr} to exist",
            )

    def test_validates_production_file(self):
        """Test must validate the actual production file, not a copy."""
        # Ensure we're testing the real file
        self.assertTrue(
            os.path.exists(self.MODULE_PATH),
            f"Production file not found at {self.MODULE_PATH}",
        )
        # Verify it's the actual file being imported
        import conversation_manager
        actual_file = os.path.abspath(conversation_manager.__file__)
        expected_file = os.path.abspath(self.MODULE_PATH)
        self.assertEqual(
            actual_file,
            expected_file,
            f"Test is not validating the production file. "
            f"Expected: {expected_file}, Got: {actual_file}",
        )


if __name__ == "__main__":
    unittest.main()
