import os

BASE_DIR = r"c:\Users\milin\Downloads\Hack Health"
DATA_FILE = os.path.join(BASE_DIR, "Primary_Emotions.xlsx - Sheet1.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "emotion_classifier", "outputs")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

SEED = 42
