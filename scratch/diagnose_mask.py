import pickle
import os
import sys
import numpy as np

pkl_path = r'd:\vt_application_v6.2_cpu\data\face_encodings.pkl'
if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Total registered faces: {len(data)}")
    for pid, info in data.items():
        name = info.get('name', 'N/A')
        has_std = info.get('encoding') is not None
        has_mask = info.get('mask_encoding') is not None
        print(f"  {pid}: {name} | std_encoding={has_std} | mask_encoding={has_mask}")
        if pid == 'PI0555':
            print(f"    >>> PI0555 DETAIL:")
            print(f"        Standard encoding shape: {np.array(info['encoding']).shape if has_std else 'NONE'}")
            if has_mask:
                print(f"        Mask encoding shape: {np.array(info['mask_encoding']).shape}")
            else:
                print(f"        Mask encoding: MISSING")
else:
    print(f"ERROR: .pkl file not found at {pkl_path}")
