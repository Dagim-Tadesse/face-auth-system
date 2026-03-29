import cv2
import pandas as pd # Great for managing the user list

def register_user(name):
    # 1. Assign a unique ID
    user_id = get_next_id() 
    
    # 2. Capture images (Your existing code here)
    capture_images(user_id)
    
    # 3. Log the user in a CSV file
    new_entry = {"ID": user_id, "Name": name, "Joined": "2026-03-29"}
    save_to_csv(new_entry)
    
    # 4. Trigger the Trainer
    train_model()
    
    print(f"Registration complete for {name} (ID: {user_id})")
