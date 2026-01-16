
import requests
import sys
import time

BASE_URL = "http://127.0.0.1:8000"

def test_endpoints():
    print(f"Testing connectivity to {BASE_URL}...")
    
    # 1. Test /docs to see if server is up at all
    try:
        r = requests.get(f"{BASE_URL}/docs")
        print(f"GET /docs status: {r.status_code}")
    except Exception as e:
        print(f"Failed to connect to {BASE_URL}: {e}")
        return

    # 2. Test /api/questions
    try:
        r = requests.get(f"{BASE_URL}/api/questions")
        print(f"GET /api/questions status: {r.status_code}")
        if r.status_code == 200:
            print("Response:", r.json())
        else:
            print("Response text:", r.text)
    except Exception as e:
        print(f"Error hitting /api/questions: {e}")

    # 3. Test /api/upload/batch-grade (expected 422 if file missing, or 404 if route missing)
    try:
        # We won't send a file, just check if route exists. 
        # If route exists but file missing -> 422 Unprocessable Entity
        # If route missing -> 404 Not Found
        r = requests.post(f"{BASE_URL}/api/upload/batch-grade")
        print(f"POST /api/upload/batch-grade status: {r.status_code} (Expected 422 if route exists, 404 if missing)")
    except Exception as e:
        print(f"Error hitting /api/upload/batch-grade: {e}")

if __name__ == "__main__":
    test_endpoints()
