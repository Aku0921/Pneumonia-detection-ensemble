#!/usr/bin/env python
"""
Test script for Pneumonia Detection API.
Run: python scripts/test_api.py
"""

import os
import sys
import requests
import time
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
END = "\033[0m"

API_URL = "http://localhost:8000"
TEST_IMAGE_PATH = Path("data/chest_xray/test/NORMAL/IM-0001-0001.jpeg")


def print_section(title):
    print(f"\n{BLUE}{'='*50}{END}")
    print(f"{BLUE}{title}{END}")
    print(f"{BLUE}{'='*50}{END}")


def check_health():
    """Check API health."""
    print_section("Health Check")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"{GREEN}✓ API is healthy{END}")
            print(f"  Model Version: {data.get('model_version')}")
            print(f"  Model Loaded: {data.get('model_loaded')}")
            print(f"  Load Time: {data.get('model_load_time_s'):.2f}s")
            return True
        else:
            print(f"{RED}✗ Health check failed (status {response.status_code}){END}")
            return False
    except Exception as e:
        print(f"{RED}✗ Cannot reach API: {e}{END}")
        return False


def test_predict():
    """Test prediction endpoint."""
    print_section("Prediction Test")
    
    if not TEST_IMAGE_PATH.exists():
        print(f"{RED}✗ Test image not found: {TEST_IMAGE_PATH}{END}")
        return False
    
    try:
        with open(TEST_IMAGE_PATH, "rb") as f:
            files = {"file": (TEST_IMAGE_PATH.name, f, "image/jpeg")}
            print(f"Uploading: {TEST_IMAGE_PATH.name}")
            
            start = time.time()
            response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"{GREEN}✓ Prediction successful ({elapsed:.2f}s){END}")
            print(f"  Prediction: {data['prediction'].upper()}")
            print(f"  Pneumonia Prob: {data['pneumonia_probability']*100:.1f}%")
            print(f"  Confidence: {data['confidence']*100:.1f}%")
            print(f"  Inference Time: {data['inference_time_ms']:.1f}ms")
            print(f"  Model Version: v{data['model_version']}")
            return True
        else:
            print(f"{RED}✗ Prediction failed (status {response.status_code}){END}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"{RED}✗ Prediction error: {e}{END}")
        return False


def test_invalid_file():
    """Test error handling with invalid file."""
    print_section("Error Handling Test")
    try:
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"{GREEN}✓ Correctly rejected invalid file{END}")
            print(f"  Status: {response.status_code}")
            print(f"  Error: {response.json().get('detail')}")
            return True
        else:
            print(f"{RED}✗ Should have rejected invalid file{END}")
            return False
    except Exception as e:
        print(f"{RED}✗ Error test failed: {e}{END}")
        return False


def main():
    print(f"{BLUE}Pneumonia Detection API Test Suite{END}")
    print(f"Testing endpoint: {BLUE}{API_URL}{END}\n")
    
    results = {
        "Health Check": check_health(),
        "Prediction": test_predict(),
        "Error Handling": test_invalid_file(),
    }
    
    print_section("Test Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = f"{GREEN}PASS{END}" if result else f"{RED}FAIL{END}"
        print(f"  {test}: {status}")
    
    print(f"\n  {BLUE}Passed: {passed}/{total}{END}")
    
    if passed == total:
        print(f"\n{GREEN}✓ All tests passed!{END}\n")
        return 0
    else:
        print(f"\n{RED}✗ Some tests failed{END}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
