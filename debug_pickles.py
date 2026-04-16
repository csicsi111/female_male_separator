import pickle
import os
from pathlib import Path

PICKLES_DIR = "downloaded_pickles"

# Get first pickle file
pkl_files = [f for f in os.listdir(PICKLES_DIR) if f.lower().endswith('.pkl')]

if pkl_files:
    pkl_path = os.path.join(PICKLES_DIR, pkl_files[0])
    print(f"Loading: {pkl_files[0]}\n")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data)}")
    print(f"Length: {len(data)}")
    
    # Check first item
    first = data[0]
    print(f"\nFirst item type: {type(first)}")
    print(f"First item dir: {[x for x in dir(first) if not x.startswith('_')]}")
    
    if hasattr(first, 'bbox'):
        bbox = first.bbox
        print(f"\nbbox type: {type(bbox)}")
        print(f"bbox value: {bbox}")
        print(f"bbox dir: {[x for x in dir(bbox) if not x.startswith('_')]}")
        print(f"isinstance(bbox, tuple): {isinstance(bbox, tuple)}")
        print(f"len(bbox): {len(bbox) if hasattr(bbox, '__len__') else 'N/A'}")
        
        # Try to convert to list
        try:
            bbox_list = list(bbox)
            print(f"bbox as list: {bbox_list}")
        except Exception as e:
            print(f"Error converting to list: {e}")
        
        # Check if it's iterable
        try:
            for i, val in enumerate(bbox):
                print(f"  Item {i}: {val} (type: {type(val)})")
                if i >= 4:
                    break
        except Exception as e:
            print(f"Error iterating: {e}")

