#!/usr/bin/env python3
"""
Simple test to verify the bug fix is working
"""

import sys
import os

print("Starting simple bug fix verification...")

try:
    print("Testing imports...")
    from jarowinkler import jarowinkler_similarity

    print("✅ jarowinkler import successful")

    # Test the similarity function
    score = jarowinkler_similarity("test", "test")
    print(f"✅ jarowinkler_similarity works: {score}")

except Exception as e:
    print(f"❌ Error with jarowinkler: {e}")
    sys.exit(1)

try:
    print("\nTesting conversation_manager imports...")
    # Import just the constants first
    from conversation_manager import (
        DEFAULT_FALLBACK_OUTPUT_TYPE,
        MINIMUM_SIMILARITY_THRESHOLD,
    )

    print(f"✅ Constants imported successfully")
    print(f"   DEFAULT_FALLBACK_OUTPUT_TYPE: '{DEFAULT_FALLBACK_OUTPUT_TYPE}'")
    print(f"   MINIMUM_SIMILARITY_THRESHOLD: {MINIMUM_SIMILARITY_THRESHOLD}")

except Exception as e:
    print(f"❌ Error importing constants: {e}")
    print(f"   Error type: {type(e)}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("✅ All basic imports successful!")
print("Bug fix verification complete.")
