from datetime import datetime
import time

class EmergencyManager:
    def __init__(self, db):
        self.db = db
        # If 'emergency_contacts' collection doesn't exist, it will be created on insert
        self.contacts = db['emergency_contacts']
        self.active_alert = None
        
    def get_contacts(self):
        """Returns list of all emergency contacts"""
        return list(self.contacts.find())

    def add_contact(self, name, phone, relation):
        """Adds a new contact"""
        contact = {
            "name": name,
            "phone": phone,
            "relation": relation,
            "created_at": datetime.now()
        }
        self.contacts.insert_one(contact)
        return True

    def delete_contact(self, contact_id):
        """Deletes a contact by ID (stored in 'created_at' or fake ID for JsonDB)"""
        # JsonDB uses internal ID, but for simplicity we might match by name/phone in this demo
        # or rely on the UI passing the right identifier.
        # For robustness with JsonDB, we'll try to find by some unique prop given we don't present IDs in UI yet.
        # Let's assume ID is passed as string.
        pass # To be implemented with UI integration

    def trigger_emergency(self, threat_type="Weapon"):
        """
        Triggers the Alert State.
        Returns the alert details to be consumed by the UI.
        """
        # debounce: don't re-trigger if already active recently (e.g. within 10s)
        if self.active_alert and (time.time() - self.active_alert['timestamp']) < 10:
            return self.active_alert

        # Find who to call
        # Logic: Call "Security" first, then "Boss"
        contact_list = self.get_contacts()
        target = contact_list[0] if contact_list else {"name": "Emergency Services", "phone": "911"}

        self.active_alert = {
            "active": True,
            "threat": threat_type,
            "calling": target['name'],
            "phone": target['phone'],
            "timestamp": time.time(),
            "message": f"DIALING {target['name']} ({target['phone']})..."
        }
        print(f"!!! EMERGENCY: {threat_type} DETECTED. DIALING {target['name']} !!!")
        return self.active_alert

    def get_status(self):
        """Returns current alert status. Auto-clears after 15 seconds."""
        if self.active_alert:
            if (time.time() - self.active_alert['timestamp']) > 5:
                # Still active but maybe change message to "Connected"
                self.active_alert['message'] = "CALL CONNECTED - ALERTING SUSPECT DETECTED"
            
            if (time.time() - self.active_alert['timestamp']) > 15:
                 self.active_alert = None # Reset
                 return {"active": False}
                 
            return self.active_alert
        return {"active": False}
