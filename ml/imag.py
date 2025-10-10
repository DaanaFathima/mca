from PIL import Image
import os
import subprocess
import shutil

def open_image_with_viewer(image_path):
    """Open image with available viewer"""
    # List of viewers to try (in order of preference)
    viewers = [
        "feh",           # Fast, lightweight
        "eog",           # GNOME default
        "gpicview",      # Very lightweight
        "gwenview",      # KDE default
        "ristretto",     # XFCE default
        "nomacs",        # Cross-platform
        "display"        # ImageMagick (if installed)
    ]
    
    for viewer in viewers:
        if shutil.which(viewer):  # Check if viewer is installed
            try:
                subprocess.Popen([viewer, image_path])
                print(f"Opened with {viewer}")
                return True
            except Exception as e:
                print(f"Failed to open with {viewer}: {e}")
                continue
    
    # Fallback to xdg-open (might still have the original issue)
    try:
        subprocess.Popen(["xdg-open", image_path])
        print("Opened with xdg-open")
        return True
    except Exception as e:
        print(f"Failed to open with xdg-open: {e}")
    
    return False

# Your main code
img = Image.open("demo.jpg")
temp_file = "temp_preview.jpg"
img.save(temp_file)

if not open_image_with_viewer(temp_file):
    print("Could not open image with any viewer")
    print("Please install an image viewer: sudo apt install feh")
