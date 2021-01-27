import sys
from pprint import pprint

from scripts import process

import cv2


path = sys.argv[-1]
image = cv2.imread(path, 0)
if image is None:
    raise FileNotFoundError

info = process(image)
pprint(info)

