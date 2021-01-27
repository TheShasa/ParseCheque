from .parser import (
    discount,
    otsearahayal
)
from .parser.base import BaseCheque
from .classificator import classification
from .cropper import crop


def process(image):
    img = crop(image)

    bank_class = classification(BaseCheque, img)

    return bank_class.parse(img)
