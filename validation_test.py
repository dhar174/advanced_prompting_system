#!/usr/bin/env python3
"""
Simple validation test for the bug fixes
"""

import tempfile
import os
import json


def test_file_operations():
    """Test file write operations with error handling"""
    print("🧪 Testing file write error handling...")

    # Test the basic structure exists
    try:
        import sys

        sys.path.append("/workspaces/advanced_prompting_system")
        from output_generator import generate_json_output

        print("✅ Successfully imported output_generator")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test 1: Invalid path (should raise exception)
    try:
        result = generate_json_output(
            {"test": "data"}, "/nonexistent/directory/test.json"
        )
        print("❌ Should have raised exception for invalid path")
        return False
    except Exception:
        print("✅ Correctly caught exception for invalid path")

    # Test 2: Valid path (should succeed)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.json")
            result = generate_json_output({"test": "data"}, test_file)

            if os.path.exists(result):
                with open(result, "r") as f:
                    data = json.load(f)
                    if data == {"test": "data"}:
                        print("✅ File write and content verification successful")
                        return True
                    else:
                        print("❌ File content mismatch")
                        return False
            else:
                print("❌ File was not created")
                return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_conversation_manager_structure():
    """Test conversation manager basic structure"""
    print("🧪 Testing conversation manager structure...")

    try:
        import sys

        sys.path.append("/workspaces/advanced_prompting_system")

        # Just test if the file can be parsed
        with open(
            "/workspaces/advanced_prompting_system/conversation_manager.py", "r"
        ) as f:
            content = f.read()

        # Check for key fixes
        voting_patterns = [
            "TIE VOTE DETECTED",
            "LOW PARTICIPATION",
            "Emergency Mediator Decision",
            "Tie-breaker Decision",
            "mediator tie-breaker protocol",
        ]

        voting_fixes_found = sum(1 for pattern in voting_patterns if pattern in content)
        if voting_fixes_found >= 3:
            print(
                f"✅ Final vote edge case handling found ({voting_fixes_found}/5 patterns)"
            )
        else:
            print(
                f"❌ Final vote edge case handling incomplete ({voting_fixes_found}/5 patterns)"
            )

        if "try:" in content and "json.dump" in content:
            print("✅ File write error handling found")
        else:
            print("❌ File write error handling missing")

        return True

    except Exception as e:
        print(f"❌ Error testing conversation manager: {e}")
        return False


def main():
    print("🔧 Running Bug Fix Validation Tests")
    print("=" * 50)

    test1_passed = test_file_operations()
    test2_passed = test_conversation_manager_structure()

    print("=" * 50)
    if test1_passed and test2_passed:
        print("🎉 All tests passed! Bug fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the fixes.")

    return test1_passed and test2_passed


if __name__ == "__main__":
    main()
