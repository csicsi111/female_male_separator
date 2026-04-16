import os
import piexif
from PIL import Image
import xml.etree.ElementTree as ET

img_path = os.path.join('downloaded_images', 'DJI_20240702102116_0044_D.jpg')

print("=" * 60)
print(f"Exploring metadata for: {os.path.basename(img_path)}")
print("=" * 60)

# Method 1: Check PIL's info
print("\nPIL Image Info:")
img = Image.open(img_path)
if hasattr(img, 'info'):
    for key, value in img.info.items():
        val_str = str(value)[:150]
        print(f"  {key}: {val_str}")

# Method 2: piexif - iterate properly
print("\nPiexif EXIF Data:")
exif_dict = piexif.load(img_path)
for ifd_name in ('0th', '1st', 'Exif', 'GPS', 'Interop', '2nd'):
    if ifd_name in exif_dict and exif_dict[ifd_name]:
        data_list = exif_dict[ifd_name]
        if isinstance(data_list, dict):
            items = data_list.items()
        else:
            items = data_list
        
        print(f"\n{ifd_name} IFD ({len(data_list)} tags):")
        for tag_data in list(items)[:20]:
            if isinstance(tag_data, tuple) and len(tag_data) == 2:
                tag, value = tag_data
            else:
                continue
            
            try:
                tag_name = piexif.TAGS[ifd_name][tag]['name']
            except:
                tag_name = f"Unknown_{tag}"
            
            # Skip long binary data
            if isinstance(value, bytes) and len(value) > 300:
                print(f"  {tag} ({tag_name}): BINARY ({len(value)} bytes)")
                # Check if it's XML (common for DJI XMP)
                try:
                    decoded = value.decode('utf-8', errors='ignore')
                    if 'http' in decoded or 'xmlns' in decoded:
                        print(f"    -> Looks like XML/XMP data")
                except:
                    pass
            else:
                val_str = str(value)[:150]
                print(f"  {tag} ({tag_name}): {val_str}")

# Method 3: Extract XMP from PIL image info
print("\n\nExtracting XMP from PIL image info:")
img = Image.open(img_path)
if 'xmp' in img.info:
    xmp_data = img.info['xmp']
    print(f"XMP data found ({len(xmp_data)} bytes)")
    try:
        xmp_str = xmp_data.decode('utf-8', errors='ignore')
        # Find gimbal-related content
        if 'Gimbal' in xmp_str or 'gimbal' in xmp_str:
            lines = xmp_str.split('\n')
            print("\nGimbal-related content:")
            for i, line in enumerate(lines):
                if 'Gimbal' in line or 'gimbal' in line or 'Yaw' in line or 'yaw' in line:
                    # Print context (line before and after)
                    if i > 0:
                        print(f"  {lines[i-1]}")
                    print(f"> {line}")
                    if i < len(lines) - 1:
                        print(f"  {lines[i+1]}")
                    print()
        else:
            print("No gimbal keywords found. First 1500 chars of XMP:")
            print(xmp_str[:1500])
    except Exception as e:
        print(f"Error decoding XMP: {e}")
