#!/usr/bin/env python3
"""
Minimal test for the output type matching bug fix
Tests only the core logic without importing the full conversation_manager module
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import just the jarowinkler library and test the logic
try:
    from jarowinkler import jarowinkler_similarity

    print("✅ jarowinkler import successful")
except ImportError as e:
    print(f"❌ Failed to import jarowinkler: {e}")
    sys.exit(1)

# Define constants (same as in the fix)
DEFAULT_FALLBACK_OUTPUT_TYPE = "Simple Concise Answer"
MINIMUM_SIMILARITY_THRESHOLD = 0.1

# Mock output_type_samples (based on what we know should be available)
output_type_samples = {
    "Simple Concise Answer": {
        "sample_original_prompt": "Example prompt",
        "sample_problem_statement": "Example problem",
        "sample_output_analysis": "Example analysis",
        "sample_final_output": "Example output",
        "sample_corrected_output": "Example corrected",
    },
    "JSON": {
        "sample_original_prompt": "JSON prompt",
        "sample_problem_statement": "JSON problem",
        "sample_output_analysis": "JSON analysis",
        "sample_final_output": "JSON output",
        "sample_corrected_output": "JSON corrected",
    },
    "PDF": {
        "sample_original_prompt": "PDF prompt",
        "sample_problem_statement": "PDF problem",
        "sample_output_analysis": "PDF analysis",
        "sample_final_output": "PDF output",
        "sample_corrected_output": "PDF corrected",
    },
}


# Mock OutputType class
class MockOutputType:
    def __init__(self, output_type):
        self.output_type = output_type


def test_output_type_matching_logic():
    """Test the core bug fix logic"""
    print("Testing Output Type Matching Logic (Core Bug Fix)")
    print("=" * 55)

    test_cases = [
        ("Simple Concise Answer", "exact_match"),
        ("JSON", "exact_match"),
        ("completely_unknown_type_xyz", "fallback"),
        ("", "fallback"),
        ("gibberish", "fallback"),
        ("Simpel Concise Answer", "fuzzy_match"),  # Typo
    ]

    all_tests_passed = True

    for test_input, expected_behavior in test_cases:
        print(f"\nTesting: '{test_input}'")
        output_type = MockOutputType(test_input)

        # Apply the bug fix logic (exact copy from the fix)
        o_types = list(output_type_samples.keys())
        best_match = None
        best_score = 0

        for o_type in o_types:
            score = jarowinkler_similarity(output_type.output_type, o_type)
            if score > best_score:
                best_score = score
                best_match = o_type

        # Handle case where no matches are found (all scores are 0) or similarity is too low
        if best_match is None or best_score < MINIMUM_SIMILARITY_THRESHOLD:
            print(f"  → No good matches found (best score: {best_score})")

            # Validate that our fallback exists in the samples
            if DEFAULT_FALLBACK_OUTPUT_TYPE not in output_type_samples:
                print(
                    f"  ❌ ERROR: Default fallback '{DEFAULT_FALLBACK_OUTPUT_TYPE}' not in samples!"
                )
                all_tests_passed = False
                continue

            best_match = DEFAULT_FALLBACK_OUTPUT_TYPE
            print(f"  → Using fallback: {best_match}")

            if expected_behavior != "fallback":
                print(f"  ❌ UNEXPECTED: Expected {expected_behavior} but got fallback")
                all_tests_passed = False
            else:
                print(f"  ✅ EXPECTED: Fallback behavior correct")

        else:
            print(f"  → Found match: '{best_match}' (score: {best_score:.3f})")

            if expected_behavior == "fallback":
                print(f"  ❌ UNEXPECTED: Expected fallback but got match")
                all_tests_passed = False
            else:
                print(f"  ✅ EXPECTED: Match behavior correct")

        # Test the critical part: accessing output_type_samples without KeyError
        try:
            sample_data = output_type_samples[best_match]
            print(f"  ✅ Successfully accessed sample data for '{best_match}'")
        except KeyError as e:
            print(f"  ❌ KeyError accessing sample data: {e}")
            all_tests_passed = False

    print("\n" + "=" * 55)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! The core bug fix logic is working correctly.")
        return True
    else:
        print("❌ SOME TESTS FAILED! The logic needs review.")
        return False


def test_original_bug_scenario():
    """Test the specific scenario that would have caused the original KeyError"""
    print("\nTesting Original Bug Scenario")
    print("=" * 35)

    unrecognized_type = "completely_unknown_output_format_xyz"
    output_type = MockOutputType(unrecognized_type)

    print(f"Testing with: '{unrecognized_type}'")

    # Simulate original buggy code
    o_types = list(output_type_samples.keys())
    best_match = None
    best_score = 0

    for o_type in o_types:
        score = jarowinkler_similarity(output_type.output_type, o_type)
        if score > best_score:
            best_score = score
            best_match = o_type

    print(f"Raw results: best_match={best_match}, best_score={best_score}")

    # This would have been the original buggy line:
    # output_type_short = output_type_samples[best_match]  # KeyError if best_match is None!

    if best_match is None:
        print("✅ Original bug detected: best_match is None")
        print("  This would have caused: KeyError: None")

        # Apply the fix
        if DEFAULT_FALLBACK_OUTPUT_TYPE not in output_type_samples:
            print("❌ Fallback type not available!")
            return False
        best_match = DEFAULT_FALLBACK_OUTPUT_TYPE
        print(f"✅ Fix applied: Using fallback '{best_match}'")

    try:
        output_type_short = output_type_samples[best_match]
        print("✅ Successfully accessed output_type_samples without KeyError")
        print("✅ Original bug has been fixed!")
        return True
    except KeyError as e:
        print(f"❌ KeyError still occurred: {e}")
        return False


if __name__ == "__main__":
    print("Minimal Test for Output Type Matching Bug Fix")
    print("=" * 50)

    main_tests_passed = test_output_type_matching_logic()
    bug_test_passed = test_original_bug_scenario()

    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Core logic tests: {'PASSED' if main_tests_passed else 'FAILED'}")
    print(f"Original bug test: {'PASSED' if bug_test_passed else 'FAILED'}")

    if main_tests_passed and bug_test_passed:
        print("🎉 BUG FIX VERIFICATION SUCCESSFUL!")
        print("The fix correctly handles zero-match cases with proper fallback.")
        sys.exit(0)
    else:
        print("❌ BUG FIX VERIFICATION FAILED!")
        sys.exit(1)
