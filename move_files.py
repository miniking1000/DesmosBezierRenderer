import os
import shutil
import time
#seeing this, it was probably written by GPT. I don't remember tho
SOURCE_DIR = r"C:\Users\vane4\Downloads"
DEST_DIR = os.path.join(SOURCE_DIR, "framesss")

os.makedirs(DEST_DIR, exist_ok=True)

def move_matching_frames():
    while True:
        for fname in os.listdir(SOURCE_DIR):
            if fname.startswith("frame-") and fname.endswith(".png"):
                src = os.path.join(SOURCE_DIR, fname)
                dest = os.path.join(DEST_DIR, fname)
                if not os.path.exists(dest):
                    try:
                        shutil.move(src, dest)
                        print(f"[OK] Moved: {fname}")
                    except Exception as e:
                        print(f"[ERROR] Couldn't move {fname}: {e}")
        time.sleep(1)

if __name__ == "__main__":
    print("📂 Watching for downloaded frames...")
    move_matching_frames()

