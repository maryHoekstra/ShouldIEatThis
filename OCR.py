# pytesseract install: https://github.com/madmaze/pytesseract

from PIL import Image
from pytesseract import image_to_string
import argparse

def main(image_path):

    # get text in image
    ingredients = image_to_string(Image.open(image_path))
    print(ingredients)


if __name__ == "__main__":

    # provide image path
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='Image path', required=True)
    args = parser.parse_args()

    main(args.image)
