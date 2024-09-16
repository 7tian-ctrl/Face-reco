import glob
import argparse
import cv2
import time
from rich import print
from rich.progress import Progress
import os
import pickle
import dlib as dl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from pathlib import Path
from collections import Counter

def get_image_files(path):
    image_files = []
    if os.path.isfile(path):
        # If the path is a file, check if it's an image file
        if path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            image_files.append(path)
    elif os.path.isdir(path):
        # If the path is a folder, recursively go through all files and subfolders
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    image_files.append(os.path.join(root, file))
    return image_files

#helper functions
def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]


BOUNDING_BOX_COLOR = "green"
TEXT_COLOR = "green"

# ...

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)

    font_size = 24
    font = ImageFont.truetype("arial.ttf", font_size)

    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )

    padding = 5
    text_left -= padding
    text_right += padding
    text_bottom += padding

    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="black",
        outline="black",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
    )

def image_to_jpg():
    root_dir = Path.cwd()
    image_files = [file for file in Path(root_dir).rglob("**/*") if file.is_file() and file.suffix.lower() in ['.png', '.gif', '.bmp', '.tiff', '.webp']]

    if len(image_files) != 0:
        try:
            with Progress() as progress:
                task = progress.add_task("[green]Converting images...", total=len(image_files))

                for image_file in image_files:
                    img = Image.open(image_file)
                    img.convert("RGB").save(image_file.with_suffix(".jpg"), "JPEG")
            
                    print(f"{image_file} converted into .jpg")

                    os.remove(image_file)
                    print(f"{image_file} has been deleted")

                    progress.update(task, advance=1)

                    time.sleep(0.1)
    
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        pass

#_______________________
#command-line-interface

parser = argparse.ArgumentParser(description="Recognize faces in an image")

parser.add_argument("--train", action="store", help="Train on input data, add path to the folder containing the images")

parser.add_argument(
    "--validate", action="store", help="Validate trained model, enter a folder path containg the images"
)

parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)

parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)   

parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)

args = parser.parse_args()

#_______________________
#main codebase


DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")


Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:

    image_to_jpg()

    names = []
    encodings = []


    for filepath in Path("training").glob("*.jpg"):
        
        print(f"Processing file: {filepath}")
        
        name = filepath.stem

        image = face_recognition.load_image_file(filepath)


        face_locations = face_recognition.face_locations(image, model=model)
        
        face_encodings = face_recognition.face_encodings(image, face_locations)
       
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
    
    
    name_encodings = {"names": names, "encodings": encodings}
    
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:

    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)

    draw = ImageDraw.Draw(pillow_image)


    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        
        _display_face(draw, bounding_box, name=name)

    del draw
    pillow_image.show()

def validate(model: str = "hog"):

    for filepath in Path("validation").rglob("*.jpg"):
        
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )


def main():
        
    if args.train:
        image_to_jpg()
        image_files = glob.glob(os.path.join(args.train, "*.jpg"))
        encode_known_faces(model=args.m)
    
    if args.validate:
        image_files = glob.glob(os.path.join(args.validate, "*.jpg"))
        validate(model=args.m)
    
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)

if __name__ == "__main__":
    main()
