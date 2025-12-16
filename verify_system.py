import requests
import time
import threading
import sys

def check_endpoints():
    base_url = "http://127.0.0.1:5000"
    print(f"Checking endpoints at {base_url}...")
    
    try:
        # 1. Check Homepage
        resp = requests.get(f"{base_url}/")
        print(f"Homepage (/): {resp.status_code}")
        
        # 2. Check Stats API
        resp = requests.get(f"{base_url}/api/stats")
        print(f"Stats API (/api/stats): {resp.status_code}")
        
        # 3. Check Emergency Status
        resp = requests.get(f"{base_url}/api/emergency_status")
        print(f"Emergency Status (/api/emergency_status): {resp.status_code}")
        
        # 4. Check Simulation
        print("Triggering Simulation...")
        resp = requests.post(f"{base_url}/api/simulate_threat", json={"type": "TEST_VERIFY"})
        print(f"Simulation Response: {resp.json()}")
        
        print("\n--- VERIFICATION COMPLETE ---")
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    # Give the server a moment to start if run in parallel (manual step usually)
    check_endpoints()
