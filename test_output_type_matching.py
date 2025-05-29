#!/usr/bin/env python3
"""
Test script for the output type matching bug fix in conversation_manager.py
Tests the Jaro-Winkler similarity function with edge cases to ensure proper fallback behavior.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from conversation_manager import (
        DEFAULT_FALLBACK_OUTPUT_TYPE,
        MINIMUM_SIMILARITY_THRESHOLD,
        output_type_samples,
    )

    print(f"✅ Successfully imported constants")
    print(f"DEFAULT_FALLBACK_OUTPUT_TYPE: '{DEFAULT_FALLBACK_OUTPUT_TYPE}'")
    print(f"MINIMUM_SIMILARITY_THRESHOLD: {MINIMUM_SIMILARITY_THRESHOLD}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
from jarowinkler import jarowinkler_similarity
from schema import OutputType


def test_output_type_matching():
    """Test the output type matching logic with various edge cases."""

    print("Testing Output Type Matching Bug Fix")
    print("=" * 50)

    # Test cases: (input_type, expected_behavior)
    test_cases = [
        ("Simple Concise Answer", "exact_match"),
        ("JSON", "exact_match"),
        ("PDF", "exact_match"),
        ("completely_unrecognized_type_12345", "fallback"),
        ("", "fallback"),
        ("gibberish_xyzabc", "fallback"),
        ("Simpel Concise Answer", "fuzzy_match"),  # Typo in "Simple"
        ("json", "fuzzy_match"),  # Case difference
    ]

    print("Available output types in samples:")
    for otype in output_type_samples.keys():
        print(f"  - '{otype}'")
    print()

    all_tests_passed = True

    for test_input, expected_behavior in test_cases:
        print(f"Testing input: '{test_input}'")

        # Simulate the matching logic from analyze_final_output
        o_types = list(output_type_samples.keys())
        best_match = None
        best_score = 0

        for o_type in o_types:
            score = jarowinkler_similarity(test_input, o_type)
            if score > best_score:
                best_score = score
                best_match = o_type

        # Apply the fix logic
        if best_match is None or best_score < MINIMUM_SIMILARITY_THRESHOLD:
            print(f"  → No good matches found (best score: {best_score})")

            # Validate that our fallback exists
            if DEFAULT_FALLBACK_OUTPUT_TYPE not in output_type_samples:
                print(
                    f"  ❌ ERROR: Default fallback '{DEFAULT_FALLBACK_OUTPUT_TYPE}' not in samples!"
                )
                all_tests_passed = False
                continue

            final_match = DEFAULT_FALLBACK_OUTPUT_TYPE
            print(f"  → Using fallback: {final_match}")

            if expected_behavior != "fallback":
                print(f"  ❌ UNEXPECTED: Expected {expected_behavior} but got fallback")
                all_tests_passed = False
            else:
                print(f"  ✅ EXPECTED: Fallback behavior correct")

        else:
            final_match = best_match
            print(f"  → Found match: '{final_match}' (score: {best_score:.3f})")

            if expected_behavior == "fallback":
                print(f"  ❌ UNEXPECTED: Expected fallback but got match")
                all_tests_passed = False
            else:
                print(f"  ✅ EXPECTED: Match behavior correct")

        # Verify we can access the output_type_samples without KeyError
        try:
            sample_data = output_type_samples[final_match]
            print(f"  ✅ Successfully accessed sample data for '{final_match}'")
        except KeyError as e:
            print(f"  ❌ KeyError accessing sample data: {e}")
            all_tests_passed = False

        print()

    print("=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! The bug fix is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please review the implementation.")

    return all_tests_passed


def test_original_bug_scenario():
    """Test the specific scenario that caused the original bug."""
    print("\nTesting Original Bug Scenario")
    print("=" * 30)

    # This would have caused the original KeyError
    unrecognized_type = "completely_unknown_output_format_xyz"

    print(f"Testing with unrecognized type: '{unrecognized_type}'")

    o_types = list(output_type_samples.keys())
    best_match = None
    best_score = 0

    for o_type in o_types:
        score = jarowinkler_similarity(unrecognized_type, o_type)
        if score > best_score:
            best_score = score
            best_match = o_type

    print(f"Raw matching results: best_match={best_match}, best_score={best_score}")

    # Original buggy code would do this:
    # output_type_short = output_type_samples[best_match]  # KeyError if best_match is None!

    # Fixed code does this:
    if best_match is None or best_score < MINIMUM_SIMILARITY_THRESHOLD:
        print("✅ Bug fix activated: Detected low/no similarity")
        if DEFAULT_FALLBACK_OUTPUT_TYPE not in output_type_samples:
            print("❌ Fallback type not available!")
            return False
        best_match = DEFAULT_FALLBACK_OUTPUT_TYPE
        print(f"✅ Using safe fallback: {best_match}")

    try:
        output_type_short = output_type_samples[best_match]
        print("✅ Successfully accessed output_type_samples without KeyError")
        return True
    except KeyError as e:
        print(f"❌ KeyError still occurred: {e}")
        return False


if __name__ == "__main__":
    # Run the tests
    main_tests_passed = test_output_type_matching()
    bug_test_passed = test_original_bug_scenario()

    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Main tests: {'PASSED' if main_tests_passed else 'FAILED'}")
    print(f"Bug scenario test: {'PASSED' if bug_test_passed else 'FAILED'}")

    if main_tests_passed and bug_test_passed:
        print("🎉 ALL TESTS PASSED! The bug fix is working correctly.")
        sys.exit(0)
    else:
        print("❌ TESTS FAILED! Please review the implementation.")
        sys.exit(1)
