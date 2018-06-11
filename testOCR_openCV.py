from PIL import Image
import pytesseract
import argparse
import cv2
import os

image = cv2.imread("images/clear_ingredient_list.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

preprocess = 'blur'
if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# make a check to see if median blurring should be done to remove
# noise
elif preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)


filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

from IPython.display import Image
Image(filename="images/clear_ingredient_list.jpg")


from PIL import Image
text = pytesseract.image_to_string(Image.open("images/clear_ingredient_list.jpg"))
os.remove(filename)
print(text)
