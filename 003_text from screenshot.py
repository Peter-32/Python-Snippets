import re
import pyautogui as g
from time import sleep
import pyperclip as clip
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn import tree
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
g.FAILSAFE = True
sleep(2)
g.size()
g.position()


def c():
    g.mouseDown()
    g.mouseUp()

def c2():
    c()
    c()

def rc():
    g.click(button='right')

def select_all_and_copy():
    rc()
    g.keyDown('a')
    g.keyUp('a')
    g.keyDown('ctrl')
    g.keyDown('c')
    g.keyUp('c')
    g.keyUp('ctrl')




im2 = g.screenshot('output/labels3/{}.png'.format(str(i).zfill(4)))


from PIL import Image

def crop(image_path, coords, saved_location):
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)


for i in range(4410):
    image = 'output/labels3/{}.png'.format(str(i).zfill(4))
    crop(image, (300,817,900,860), 'output/cropped_labels3/{}.png'.format(str(i).zfill(4)))


    pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/4.0.0_1/bin/tesseract'

labels3 = []
for i in range(4410):
    s = pytesseract.image_to_string(Image.open('output/cropped_labels3/{}.png'.format(str(i).zfill(4))))
    match1 = re.match("(.*?\d+?.\d) ", s)
    try:
        number = float(match1.group(1).strip())
        labels3.append(number)
    except:
        labels3.append(None)


for i, label in zip(range(len(labels3)), labels3):
    print(i, label)
