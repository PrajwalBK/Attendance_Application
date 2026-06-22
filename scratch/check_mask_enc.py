import pickle, numpy as np
pkl = r'd:\vt_application_v6.2_cpu\data\face_encodings.pkl'
with open(pkl, 'rb') as f:
    data = pickle.load(f)
print(f"Total registered faces: {len(data)}")
for pid, info in data.items():
    name = info.get('name', 'N/A')
    has_std = info.get('encoding') is not None
    has_mask = info.get('mask_encoding') is not None
    print(f"  {pid}: {name} | std={has_std} | mask={has_mask}")
