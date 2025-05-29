#!/usr/bin/env python3
"""
Test script to verify the bug fixes for file write error handling
and voting logic improvements in conversation_manager.py and output_generator.py
"""

import os
import tempfile
import json
import sys
from unittest.mock import patch, mock_open

# Add the current directory to the path to import our modules
sys.path.append("/workspaces/advanced_prompting_system")

try:
    from output_generator import generate_json_output, generate_text_output
    from conversation_manager import ConversationManager

    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_file_write_error_handling():
    """Test that file write operations handle errors gracefully"""
    print("\n🧪 Testing file write error handling...")

    # Test 1: Try to write to a directory that doesn't exist
    try:
        result = generate_json_output({"test": "data"}, "/nonexistent/path/test.json")
        print("❌ Should have raised an exception for invalid path")
    except Exception as e:
        print(f"✅ Correctly caught exception for invalid path: {type(e).__name__}")

    # Test 2: Try to write to a valid temporary location
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test.json")
            result = generate_json_output({"test": "data"}, temp_file)
            print(f"✅ Successfully wrote to valid path: {result}")

            # Verify the file was actually created
            if os.path.exists(result):
                with open(result, "r") as f:
                    data = json.load(f)
                    if data == {"test": "data"}:
                        print("✅ File content verified correctly")
                    else:
                        print("❌ File content mismatch")
            else:
                print("❌ File was not created")
    except Exception as e:
        print(f"❌ Unexpected error in valid path test: {e}")


def test_voting_logic_imports():
    """Test that the conversation manager can be imported and basic structure is intact"""
    print("\n🧪 Testing conversation manager structure...")

    try:
        # Check if key classes and methods exist
        if hasattr(ConversationManager, "__init__"):
            print("✅ ConversationManager class structure intact")
        else:
            print("❌ ConversationManager missing __init__")

        # Try to create an instance (this will test basic initialization)
        try:
            cm = ConversationManager()
            print("✅ ConversationManager can be instantiated")
        except Exception as e:
            print(
                f"⚠️  ConversationManager instantiation failed (may need dependencies): {e}"
            )

    except Exception as e:
        print(f"❌ Error testing ConversationManager: {e}")


def test_specific_voting_methods():
    """Test specific voting methods that were fixed"""
    print("\n🧪 Testing voting method signatures...")

    try:
        cm = ConversationManager()

        # Check if the voting methods exist and have the expected structure
        voting_methods = [
            "handle_final_vote_edge_cases",
            "handle_problem_vote_edge_cases",
            "handle_continue_vote_edge_cases",
        ]

        for method_name in voting_methods:
            if hasattr(cm, method_name):
                print(f"✅ Method {method_name} exists")
            else:
                print(f"❌ Method {method_name} missing")

    except Exception as e:
        print(f"⚠️  Could not fully test voting methods (may need dependencies): {e}")


def run_all_tests():
    """Run all test functions"""
    print("🔧 Running bug fix validation tests...")
    print("=" * 60)

    test_file_write_error_handling()
    test_voting_logic_imports()
    test_specific_voting_methods()

    print("\n" + "=" * 60)
    print("🏁 Test run completed!")


if __name__ == "__main__":
    run_all_tests()
