import cv2
import face_recognition
import numpy as np
import os
import threading
import time
import pickle
from datetime import datetime
from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME
from pymongo import MongoClient
from ultralytics import YOLO
from emergency_manager import EmergencyManager

class CameraStream:
    def __init__(self, src, name, process_callback=None):
        self.src = src
        self.name = name
        self.process_callback = process_callback
        # Initialize
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.output_frame = None
        if self.grabbed:
            self.output_frame = self.frame.copy()

    def start(self):
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            try:
                (grabbed, frame) = self.stream.read()
                self.grabbed = grabbed
                if not grabbed:
                    # Retry logic: Re-open if possible or wait
                    time.sleep(0.5)
                    try:
                         # Attempt reconnect
                         self.stream.release()
                         self.stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
                    except: pass
                    continue
                
                # Process frame
                if self.process_callback:
                    try:
                        frame = self.process_callback(frame)
                    except Exception as e:
                        print(f"Error processing callback: {e}")
                        # If processing fails, still show raw frame!
                
                with self.read_lock:
                    self.output_frame = frame.copy()
            
            except Exception as e:
                print(f"Stream Error: {e}")
                time.sleep(0.5)

            # Cap at ~30 FPS to save CPU
            time.sleep(0.03)

    def read(self):
        with self.read_lock:
            return self.output_frame if self.output_frame is not None else None

    def stop(self):
        self.started = False
        if hasattr(self, 'thread') and self.thread.is_alive():
             self.thread.join(timeout=1.0)
        self.stream.release()

class CameraManager:
    def __init__(self, app_config, db):
        self.app_config = app_config
        self.db = db
        self.persons = self.db[COLLECTION_NAME]
        
        # Initialize Emergency Manager
        self.emergency = EmergencyManager(self.db)
        
        # Initialize YOLO
        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt') 
        # Standard COCO classes: 43=knife, 76=scissors. 
        # Extended & Proxy Classes:
        # 34=Bat, 39=Bottle (Real)
        # 65=Remote (Handgun Proxy), 25=Umbrella (Rifle Proxy)
        # 67=Cell Phone (Simulated Trigger)
        self.threat_classes = {
            43: "Knife", 76: "Scissors",
            34: "Baseball Bat", 39: "Glass Bottle",
            65: "Handgun (Glock)", 25: "Rifle (AK47/M4)", 
            67: "Simulated Trigger"
        }
        self.class_names = self.model.names

        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_relations = [] # To store relation (e.g. employee, family)
        
        # Stats
        self.stats = {
            "suspects": 0,
            "unknown": 0,
            "known": 0,
            "traffic": 0,
            "history": [] # Simple log
        }
        self.stats_lock = threading.Lock()
        
        # Auto-Registration Counter
        # Check highest "Unknown X" in DB to resume numbering
        last_unknown = self.persons.find_one({"name": {"$regex": "^Unknown \d+"}}, sort=[("created_at", -1)])
        self.auto_id_counter = 1
        if last_unknown:
            try:
                self.auto_id_counter = int(last_unknown['name'].split(" ")[1]) + 1
            except: pass
        
        # Ensure captures dir exists
        self.captures_dir = os.path.join(app_config['UPLOAD_FOLDER'], 'captures')
        os.makedirs(self.captures_dir, exist_ok=True)
        
        self.load_known_faces()

        if len(self.known_face_names) == 0:
            print(f"Warning: No known faces loaded. Detection will only show 'Unknown'.")



    def _load_cache(self):
        cache_path = "encodings_cache.pkl"
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Cache corrupted or unreadable ({e}). Ignoring.")
        return {}

    def _save_cache(self, cache):
        try:
            with open("encodings_cache.pkl", 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def load_known_faces(self):
        """Loads known faces from database with caching to improve performance."""
        print("Loading known faces...")
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_relations = []
        
        cache = self._load_cache()
        new_cache = {}
        
        all_persons = list(self.persons.find())
        
        def get_encoding(path):
            # 1. Check Cache
            if path in cache:
                new_cache[path] = cache[path]
                return cache[path]
            
            # 2. Compute
            if not os.path.exists(path): return None
            try:
                image = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(image)
                if len(encs) > 0:
                    new_cache[path] = encs[0]
                    return encs[0]
            except Exception as e:
                print(f"Error processing {path}: {e}")
            return None

        for person in all_persons:
            encodings_found = 0
            
            # 1. Try Directory
            if 'photo_dir' in person and person['photo_dir']:
                 dir_path = os.path.join(self.app_config['UPLOAD_FOLDER'], person['photo_dir'])
                 if os.path.exists(dir_path):
                     for fname in os.listdir(dir_path):
                         # skip non-images simple check
                         if not fname.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                         
                         full_path = os.path.join(dir_path, fname)
                         enc = get_encoding(full_path)
                         if enc is not None:
                             self.known_face_encodings.append(enc)
                             self.known_face_names.append(person['name'])
                             self.known_face_relations.append(person['relation'])
                             encodings_found += 1
            
            # 2. Try Single File (Legacy)
            if encodings_found == 0:
                photo_path = os.path.join(self.app_config['UPLOAD_FOLDER'], person['photo'])
                enc = get_encoding(photo_path)
                if enc is not None:
                    self.known_face_encodings.append(enc)
                    self.known_face_names.append(person['name'])
                    self.known_face_relations.append(person['relation'])

        self._save_cache(new_cache)
        print(f"Loaded {len(self.known_face_names)} faces.")

    def process_frame(self, frame):
        """Detects faces and annotates the frame"""
        # Resize frame of video to 1/4 size for faster face recognition processing
        # Ensure frame is valid
        if frame is None or frame.size == 0:
             return frame
             
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # Using numpy slicing for BGR to RGB conversion which is faster than cv2.cvtColor
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        current_frame_stats = {"known": 0, "unknown": 0, "suspects": 0}


        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
             # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            relation = "Stranger"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    relation = self.known_face_relations[best_match_index]
            
            # --- AUTO REGISTRATION FOR UKNOWNS ---
            if name == "Unknown":
                try:
                    # 1. Generate Info
                    new_name = f"Unknown {self.auto_id_counter}"
                    self.auto_id_counter += 1
                    relation = "Auto-Detected" # Mark as auto-saved
                    
                    # 2. Save Image
                    # Need to extract face *before* drawing boxes (we have coords)
                    # Coordinates are for small frame, scale up for save
                    top_s, right_s, bottom_s, left_s = top, right, bottom, left
                    top_f, right_f, bottom_f, left_f = top*4, right*4, bottom*4, left*4
                    
                    # Clamp
                    h, w, _ = frame.shape
                    top_f = max(0, top_f); left_f = max(0, left_f)
                    bottom_f = min(h, bottom_f); right_f = min(w, right_f)
                    
                    face_img_save = frame[top_f:bottom_f, left_f:right_f].copy()
                    
                    if face_img_save.size > 0:
                        filename = f"{new_name.replace(' ', '_')}.jpg"
                        # Use a dedicated folder or just uploads/known
                        save_path = os.path.join(self.app_config['UPLOAD_FOLDER'], 'known', filename)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, face_img_save)
                        
                        rel_path = f"known/{filename}"
                        
                        # 3. Save to DB
                        self.persons.insert_one({
                            "serial_no": 9000 + self.auto_id_counter, # Special range for autos
                            "name": new_name,
                            "relation": relation,
                            "phone": "N/A",
                            "address": "Auto-Captured",
                            "photo": rel_path,
                            "created_at": datetime.now()
                        })
                        
                        # 4. Update Memory (Immediate Recognition)
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(new_name)
                        self.known_face_relations.append(relation)
                        
                        # 5. Update Current
                        name = new_name
                        print(f"Auto-Registered: {name}")
                
                except Exception as e:
                    print(f"Auto-reg error: {e}")

            # Update Stats logic
            if relation == "Auto-Detected" or name.startswith("Unknown"):
                 current_frame_stats["unknown"] += 1
            else:
                 current_frame_stats["known"] += 1
                
            # --- CAPTURE SNAPSHOT ---
            # Scale up
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            # Clamp coords
            h, w, _ = frame.shape
            top = max(0, top); left = max(0, left)
            bottom = min(h, bottom); right = min(w, right)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_filename = f"event_{timestamp}_{name}.jpg"
            snap_path = os.path.join(self.captures_dir, snap_filename)
            
            # Save cropped face (only if not recently logged to avoid spamming I/O)
            # We check stats history in log_event, so maybe safe to save ?
            # Better: only save if we are GOING to log it.
            
            face_img = frame[top:bottom, left:right]
            if face_img.size > 0:
                 # Pass image to log_event to save ONLY if new event
                 # Debug: print(f"Logging {name}")
                 self.log_event(name, "Detected", relation, face_img)

            # Draw Box & Label
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            if "suspect" in relation.lower(): color = (0, 165, 255) # Orange
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{name} ({relation})", (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

            if "suspect" in relation.lower():
                 self.emergency.trigger_emergency("Known Suspect")

        # --- YOLO OBJECT DETECTION (WEAPONS) ---

        # --- YOLO OBJECT DETECTION (WEAPONS & FIGHTS) ---
        # Run inference on the small frame (RGB)
        results = self.model(rgb_small_frame, verbose=False, iou=0.5, conf=0.4)
        
        person_boxes = [] # For fight detection

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                
                # Person Detection (Class 0)
                if cls == 0:
                     x1, y1, x2, y2 = box.xyxy[0]
                     # Scale back up
                     x1, y1, x2, y2 = int(x1*4), int(y1*4), int(x2*4), int(y2*4)
                     person_boxes.append((x1, y1, x2, y2))
                     continue

                # Weapon Detection
                if cls in self.threat_classes:
                    # Detected a threat
                    x1, y1, x2, y2 = box.xyxy[0]
                    # Scale back up
                    x1, y1, x2, y2 = int(x1*4), int(y1*4), int(x2*4), int(y2*4)
                    
                    label = self.threat_classes[cls]
                    
                    # Draw RED Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"THREAT: {label}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                    self.emergency.trigger_emergency(f"Weapon ({label})")
                    with self.stats_lock:
                        self.stats["suspects"] += 1 # Count as suspect activity

        # --- FIGHT DETECTION (HEURISTIC) ---
        if len(person_boxes) >= 2:
            import itertools
            for (box1, box2) in itertools.combinations(person_boxes, 2):
                # Calculate IOU or simple intersection
                xA = max(box1[0], box2[0])
                yA = max(box1[1], box2[1])
                xB = min(box1[2], box2[2])
                yB = min(box1[3], box2[3])
                
                interArea = max(0, xB - xA) * max(0, yB - yA)
                box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                
                # Intersection over Union
                iou = interArea / float(box1Area + box2Area - interArea)
                
                # If high overlap, flag as fight
                if iou > 0.35:
                     # Draw "FIGHT" box around encompassing area
                     fx1 = min(box1[0], box2[0])
                     fy1 = min(box1[1], box2[1])
                     fx2 = max(box1[2], box2[2])
                     fy2 = max(box1[3], box2[3])
                     
                     cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (128, 0, 128), 4)
                     cv2.putText(frame, "VIOLENCE DETECTED", (fx1, fy1 - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 3)
                     
                     self.log_event("System", "Violence Detected", "Suspect")
                     self.emergency.trigger_emergency("Violence / Fighting")

        # Update global accumulative stats
        # We rely on log_event for accurate "Event" counting now.
        pass

    def log_event(self, name, action, relation="Visitor", face_img=None):
        """Adds an event to the history log"""
        # Limit log size
        if len(self.stats["history"]) > 20: # Increased log size
            self.stats["history"].pop(0)
            
        # Avoid duplicate consecutive logs (debounce 2 seconds)
        now = datetime.now()
        if self.stats["history"]:
            last = self.stats["history"][-1]
            last_time = datetime.strptime(last['time'], "%H:%M:%S")
            seconds_diff = abs((now - now.replace(hour=last_time.hour, minute=last_time.minute, second=last_time.second)).total_seconds())
            
            # Same name/action debounce
            if last['name'] == name and last['action'] == action and seconds_diff < 3: 
                return

        # Increment Stats (Event Based)
        with self.stats_lock:
            if name == "System": # System events (Weapons)
                self.stats["suspects"] += 1
            elif name == "Unknown":
                self.stats["unknown"] += 1
            else:
                self.stats["known"] += 1

        # Save Image if provided
        snap_rel_path = "default_avatar.png"
        if face_img is not None:
            clean_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
            ts = now.strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{clean_name}.jpg"
            save_path = os.path.join(self.captures_dir, filename)
            try:
                cv2.imwrite(save_path, face_img)
                # Client needs path relative to 'static'. 
                # UPLOAD_FOLDER is 'static/uploads'.
                # So we want 'uploads/captures/filename'.
                snap_rel_path = f"uploads/captures/{filename}"
            except Exception as e:
                print(f"Failed to save snap: {e}")

        self.stats["history"].append({
            "name": name,
            "action": action,
            "relation": relation,
            "image": snap_rel_path,
            "time": now.strftime("%H:%M:%S")
        })


    def get_stats(self):
        with self.stats_lock:
            return self.stats
