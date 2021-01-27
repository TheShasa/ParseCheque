from cv2 import cv2
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import time


def proccess_image(img):
    width = 1200
    height = 600
    img = cv2.resize(img, (width, height))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 3)
    return img


orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher()


class Sample:
    def __init__(self, descriptor, icon_area, bank_name):
        self.descriptor = descriptor
        self.icon_area = icon_area
        self.bank_name = bank_name

    def crop_by_icon_area(self, img):
        area = self.icon_area
        return img[area[0]:area[1], area[2]:area[3]]

    def match(self, img):
        img = proccess_image(img)
        img = self.crop_by_icon_area(img)
        _, other_des = orb.detectAndCompute(img, None)
        if other_des is None:
            return -1

        matches = bf.knnMatch(self.descriptor, other_des, k=2)
        if matches and len(matches[0]) != 2:
            return -2

        score = 0
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                score += 1
        return score

    def show_icon_area(self, img):
        plt.imshow(self.crop_by_icon_area(img))
        plt.show()


class Classificator:
    def __init__(self, icons_path='scripts/classificator/descriptors/'):
        self.examples = []
        for name in os.listdir(icons_path):
            if name[-5:] == '.json':
                with open(os.path.join(icons_path, name)) as f:
                    data = json.loads(f.read())
                    data['descriptor'] = np.matrix(data['descriptor'], dtype='uint8')
                    self.examples.append(Sample(**data))
        if not self.examples:
            raise Exception

    def match(self, img):
        best = (float('-inf'), '')
        for sample in self.examples:
            # _start = time.time()
            point = sample.match(img)
            if point > best[0]:
                best = (point, sample.bank_name)
            # print(sample.bank_name, time.time() - _start)
        return best[1]

# cf = Classificator()
# def classification(cls, image):
#     return cf.match(image)


def match_by_class(base_cls, img):
    best = (float('-inf'), base_cls)
    for cls in base_cls.__subclasses__():
        sample = Sample(descriptor=cls.descriptor,
                        icon_area=cls.icon_area,
                        bank_name=cls.TYPE_NAME)

        point = sample.match(img)
        if point > best[0]:
            best = (point, cls)
    return best[1]


classification = match_by_class
