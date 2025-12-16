import json
import os
from datetime import datetime

class JsonDB:
    """Authentication and Database mock acting as a MongoDB Database object."""
    def __init__(self, db_name="local_data"):
        self.name = db_name
        self.collections = {}

    def __getitem__(self, collection_name):
        if collection_name not in self.collections:
            self.collections[collection_name] = JsonCollection(collection_name)
        return self.collections[collection_name]

class JsonCollection:
    """Mock MongoDB Collection that persists to a specific JSON file."""
    def __init__(self, name):
        self.filename = f"{name}.json"
        self.data = []
        self._load()

    def _load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    self.data = json.load(f, object_hook=self._datetime_hook)
            except:
                self.data = []

    def _save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, default=str, indent=2)

    def _datetime_hook(self, dct):
        if 'created_at' in dct and isinstance(dct['created_at'], str):
            try:
                dct['created_at'] = datetime.fromisoformat(dct['created_at'])
            except:
                pass
        return dct

    def find(self, filter_dict=None):
        return JsonCursor(self.data, filter_dict)

    def find_one(self, filter_dict=None, sort=None):
        results = self.find(filter_dict).sort(sort)._results
        return results[0] if results else None

    def insert_one(self, doc):
        # Generate simple ID if not present
        if '_id' not in doc:
             doc['_id'] = str(int(datetime.now().timestamp() * 1000))
        self.data.append(doc)
        self._save()
        return type('obj', (object,), {'inserted_id': doc['_id']})

    def delete_one(self, filter_dict):
        target = self.find_one(filter_dict)
        if target:
            self.data.remove(target)
            self._save()

    def update_one(self, filter_dict, update_dict):
        target = self.find_one(filter_dict)
        if target and "$set" in update_dict:
            for k, v in update_dict["$set"].items():
                target[k] = v
            self._save()

class JsonCursor:
    def __init__(self, data, filter_dict):
        self._results = []
        if not filter_dict:
            self._results = data.copy()
        else:
            for item in data:
                match = True
                for k, v in filter_dict.items():
                    # Support simple equality
                    if item.get(k) != v:
                        match = False
                        break
                if match:
                    self._results.append(item)

    def sort(self, key_or_list, direction=1):
        if not self._results:
            return self
            
        key = key_or_list
        reverse = False
        
        # Handle simple tuple list format: [("key", -1)]
        if isinstance(key_or_list, list):
            key, direction = key_or_list[0]
        
        if direction == -1:
            reverse = True
            
        self._results.sort(key=lambda x: x.get(key, 0), reverse=reverse)
        return self

    def __iter__(self):
        return iter(self._results)

    def __list__(self):
        return self._results
