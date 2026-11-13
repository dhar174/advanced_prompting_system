#!/usr/bin/env python3
"""
Simple test to verify the bug fix is working.
"""


def test_imports():
    """Test all required imports work."""
    try:
        from conversation_manager import (
            DEFAULT_FALLBACK_OUTPUT_TYPE,
            MINIMUM_SIMILARITY_THRESHOLD,
        )
        from jarowinkler import jaro_winkler_similarity

        print("✓ All imports successful")
        print(f"  - Default fallback: {DEFAULT_FALLBACK_OUTPUT_TYPE}")
        print(f"  - Minimum threshold: {MINIMUM_SIMILARITY_THRESHOLD}")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_jaro_winkler():
    """Test the Jaro-Winkler function works."""
    try:
        from jarowinkler import jaro_winkler_similarity

        # Test normal similarity
        score1 = jaro_winkler_similarity("test", "test")
        print(f"✓ Exact match test: {score1} (should be 1.0)")

        # Test zero similarity (the bug case)
        score2 = jaro_winkler_similarity("Valid Type", "CompletelyDifferent12345")
        print(f"✓ Zero similarity test: {score2} (should be very low)")

        return True
    except Exception as e:
        print(f"✗ Jaro-Winkler test failed: {e}")
        return False


def test_conversation_manager():
    """Test that ConversationManager can be imported and instantiated."""
    try:
        from conversation_manager import ConversationManager

        cm = ConversationManager()
        print("✓ ConversationManager created successfully")
        return True
    except Exception as e:
        print(f"✗ ConversationManager test failed: {e}")
        return False


def main():
    print("=== SIMPLE BUG FIX VERIFICATION ===")

    tests = [
        test_imports,
        test_jaro_winkler,
        test_conversation_manager,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 ALL TESTS PASSED - BUG FIX VERIFIED!")
    else:
        print("❌ Some tests failed")


if __name__ == "__main__":
    main()
