#!/usr/bin/env python3
"""
Final verification script for the Jaro-Winkler output type matching bug fix.
"""

import sys
import traceback


def test_import():
    """Test that all required modules can be imported."""
    try:
        from conversation_manager import (
            ConversationManager,
            DEFAULT_FALLBACK_OUTPUT_TYPE,
            MINIMUM_SIMILARITY_THRESHOLD,
        )
        from jarowinkler import jaro_winkler_similarity

        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_jaro_winkler():
    """Test the Jaro-Winkler similarity function."""
    try:
        from jarowinkler import jaro_winkler_similarity

        # Test exact match
        score = jaro_winkler_similarity("test", "test")
        assert score == 1.0, f"Expected 1.0, got {score}"

        # Test no match (the bug scenario)
        score = jaro_winkler_similarity("Valid Type", "CompletelyDifferent12345")
        assert score < 0.1, f"Expected very low score, got {score}"

        print("✓ Jaro-Winkler similarity function works correctly")
        return True
    except Exception as e:
        print(f"✗ Jaro-Winkler test failed: {e}")
        traceback.print_exc()
        return False


def test_bug_fix():
    """Test the specific bug fix in analyze_final_output."""
    try:
        from conversation_manager import ConversationManager

        cm = ConversationManager()

        # Test with valid output type (should work)
        result1 = cm.analyze_final_output("Test content", "Simple Concise Answer")
        assert result1 is not None, "Valid output type should return a result"

        # Test with invalid output type (should use fallback, not crash)
        result2 = cm.analyze_final_output("Test content", "NonExistentType12345")
        assert result2 is not None, "Invalid output type should use fallback, not crash"

        print("✓ Bug fix works correctly - no crashes with invalid output types")
        return True
    except Exception as e:
        print(f"✗ Bug fix test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("=" * 50)
    print("FINAL VERIFICATION: Jaro-Winkler Bug Fix")
    print("=" * 50)

    tests = [
        ("Import Test", test_import),
        ("Jaro-Winkler Function Test", test_jaro_winkler),
        ("Bug Fix Test", test_bug_fix),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"FAILED: {test_name}")

    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - BUG FIX SUCCESSFUL!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
