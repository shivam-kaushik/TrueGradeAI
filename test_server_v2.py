
import requests
import sys

BASE_URL = "http://127.0.0.1:8001"

def run_test():
    print(f"--- STARTING CONNECTIVITY TEST TO {BASE_URL} ---")
    
    # 1. Check Root/Docs
    img_failed = False
    try:
        r = requests.get(f"{BASE_URL}/docs", timeout=2)
        print(f"[PASS] GET /docs -> {r.status_code}")
    except Exception as e:
        print(f"[FAIL] Could not connect to backend: {e}")
        return

    # 2. Check /api/questions
    try:
        r = requests.get(f"{BASE_URL}/api/questions", timeout=2)
        print(f"[{'PASS' if r.status_code == 200 else 'FAIL'}] GET /api/questions -> {r.status_code}")
        if r.status_code != 200:
            print(f"Body: {r.text}")
    except Exception as e:
        print(f"[FAIL] GET /api/questions error: {e}")

    # 3. Check /api/upload/batch-grade (POST)
    print("\n--- TESTING BATCH GRADE UPLOAD ---")
    files = {
        'file': ('mock_student.csv', open('mock_student.csv', 'rb'), 'text/csv')
    }
    
    try:
        url = f"{BASE_URL}/api/upload/batch-grade"
        print(f"POSTing to {url}...")
        r = requests.post(url, files=files, timeout=5)
        
        print(f"Status Code: {r.status_code}")
        print(f"Response Body: {r.text}")
        
        if r.status_code == 200:
            print("[SUCCESS] Batch grade endpoint is working!")
        elif r.status_code == 404:
            print("[CRITICAL FAILURE] Endpoint verification returned 404 (Not Found).")
            print("Possible causes: Route typo, trailing slash mismatch, or server not reloading.")
        else:
            print(f"[FAILURE] Unexpected status code: {r.status_code}")
            
    except Exception as e:
        print(f"[FAIL] POST error: {e}")

if __name__ == "__main__":
    run_test()
