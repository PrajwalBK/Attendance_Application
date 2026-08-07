import os
import sys
import base64
import json

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config.config import get_config
from core.api_client import APIClient

# Read and decode auth credentials from config/auth_config.json
auth_path = os.path.join("config", "auth_config.json")
if os.path.exists(auth_path):
    with open(auth_path, "r") as f:
        auth_data = json.load(f)
        
    email = base64.b64decode(auth_data.get("u")).decode("utf-8")
    password = base64.b64decode(auth_data.get("p")).decode("utf-8")
    url = base64.b64decode(auth_data.get("a")).decode("utf-8")
    
    print(f"Decoded credentials from auth_config.json:")
    print(f"  URL: {url}")
    print(f"  User: {email}")
else:
    print("Error: auth_config.json not found.")
    sys.exit(1)

try:
    print("\nInitializing APIClient and fetching face encodings...")
    client = APIClient(url, api_user=email, api_password=password)
    encodings = client.get_all_face_encodings()
    
    print(f"\nSuccessfully retrieved {len(encodings)} records from server:")
    with_mask = 0
    with_face = 0
    
    for idx, (pid, info) in enumerate(encodings.items()):
        name = info.get('name')
        has_face = info.get('encoding') is not None
        has_mask = info.get('mask_encoding') is not None
        
        if has_face: with_face += 1
        if has_mask: with_mask += 1
        
        print(f"Employee {idx+1}: ID={pid}, Name={name} | Has Standard Face Encoding={has_face} | Has Mask Encoding={has_mask}")
        if has_mask:
            mask_shape = info.get('mask_encoding').shape if hasattr(info.get('mask_encoding'), 'shape') else "N/A"
            print(f"  - Mask encoding shape: {mask_shape}")
        
    print("\n" + "="*50)
    print(f"Summary:")
    print(f"Total employees: {len(encodings)}")
    print(f"Employees with standard face encoding: {with_face}")
    print(f"Employees with mask face encoding: {with_mask}")
    print("="*50)
    
except Exception as e:
    print(f"Error executing check: {e}")
