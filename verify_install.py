import face_recognition
import numpy as np
import cv2

print("--- START TEST ---")
try:
    print("1. Importing face_recognition...")
    # This checks if dlib links correctly
    import dlib
    print(f"   dlib version: {dlib.__version__}")

    print("2. Creating dummy image...")
    img = np.zeros((300, 300, 3), dtype="uint8")
    
    print("3. Running face_locations (HOG model)...")
    locs = face_recognition.face_locations(img, model="hog")
    print(f"   Success. Found {len(locs)} faces (expected 0).")
    
    print("--- PASS ---")
except Exception as e:
    print(f"--- FAILED ---")
    print(e)
