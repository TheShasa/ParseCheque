# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def intersection(line1, line2):
    a_x1, a_y1,a_x2,a_y2 = line1
    b_x1, b_y1,b_x2,b_y2 = line2
    A1 = a_y2 - a_y1
    B1 = a_x1 - a_x2
    C1 = a_y1 * (a_x2 -a_x1)  - a_x1 * (a_y2 - a_y1)
    C1 *= -1
    A2 = b_y2 - b_y1
    B2 = b_x1 - b_x2
    C2 = b_y1 * (b_x2 -b_x1)  - b_x1 * (b_y2 - b_y1)
    C2 *= -1
    determ = A1*B2 - A2*B1

    return [(C1*B2 - C2*B1) // determ, ( C2*A1-C1*A2 ) // determ]


def get_lines(edges, thresh=220, gap=550):
    lines = cv2.HoughLinesP(edges,1,np.pi/180,thresh,maxLineGap=gap)
    length = len(lines)
    # print("number of lines :", length)
    format_line = np.zeros([length,4],dtype='int32')
    for i in range(length):
        format_line[i] =lines[i][0]
    return format_line


def vertical_and_horizontal(lines):
    ver, hor = [], []
    for line in lines:
        x1, y1, x2, y2 = line
        if abs(x2 - x1) > abs(y2 - y1):
            hor.append(line)
        else:
            ver.append(line)

    ver = np.array(sorted(ver, key=lambda l: min(l[0], l[2])))
    hor = np.array(sorted(hor, key=lambda l: min(l[1], l[3])))
    return ver, hor


def line_filter(lines):
    """:returns top, bottom, left, right"""
    ver, hor = vertical_and_horizontal(lines)

    return hor[0], hor[-1], ver[0], ver[-1]

def get_matrix(lines):
    top, bottom, left, right = line_filter(lines)

    inter = np.float32([
        intersection(top, left),
        intersection(top, right),
        intersection(bottom, left),
        intersection(bottom, right)
    ])

    width = points_distance(inter[0], inter[1])
    height = points_distance(inter[0], inter[2])
    size = np.float32([[0,0], [width, 0], [0,height], [width,height]])

    return cv2.getPerspectiveTransform(inter, size), (width, height)


def points_distance(p1, p2):
    return int(np.linalg.norm([p1[0]-p2[0], p1[1]-p2[1]]))


def before(image):
    gray = image
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    gray_after_kernel = cv2.erode(gray, kernel,iterations=3)
    edges = cv2.Canny(gray_after_kernel,75,150)
    return edges

def after(image):
    gray = image
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.blur(gray,(5,5))
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray_after_filter = cv2.filter2D(gray,-1,filter)
    edges = cv2.Canny(gray_after_filter,75,150)

    #make lines on edgers smooth
    blur = cv2.GaussianBlur( edges, (5,5), 0)
    smooth = cv2.addWeighted( blur, 1.5, edges, -0.5, 0)
    return smooth


def img_read(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_lines(image, lines):
    lined_img = image.copy()

    for line in lines:
        x1,y1,x2,y2 = line
        cv2.line(lined_img,(x1,y1),(x2,y2),(0,255,255),5)
    return lined_img


def medium_in_histogram(image_gray) -> int:
    return int(image_gray.mean())


def make_threshold(image_gray):
    shape = np.max(image_gray.shape)//80
    shape += not shape % 2

    blur = cv2.GaussianBlur(image_gray, (shape,shape), shape//3)

    medium = medium_in_histogram(image_gray)
    _, threshold = cv2.threshold(blur, medium, 255, cv2.THRESH_BINARY)
    kernal = np.ones((15, 15), np.uint8)
    threshold = cv2.erode(threshold, kernal)

    return after(threshold)


def biggest_contour(image_threshold, with_convex=False):
    contours, _ = cv2.findContours(image_threshold,
                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_sizes = ((cv2.contourArea(contour), contour) for contour in contours)
    cnt = max(contour_sizes, key=lambda _: _[0])[1]

    if with_convex:
        cnt = cv2.convexHull(cnt)

    return cnt


def rotate_to_horizontal(image):
    height, width = image.shape[:2]
    if width < height:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


def is_gray(image):
    return len(image.shape) < 3 or image.shape[2] == 1


def concatenate_images(*images):
    height, width = images[0].shape[:2]

    method = cv2.hconcat if width < height else cv2.vconcat

    images = list(images)
    for i, image in enumerate(images):
        if is_gray(image):
            images[i] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB, dst=image)

    return method(images)


class Cropper:
    def __init__(self, image):
        self.img = image

        try:
            self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except cv2.error:
            self.gray = image

    def crop(self, show_result=False, show_histogram=False, show_lines=False):
        imgs_to_show = []

        th = make_threshold(self.gray)

        show_cnt = self.img.copy() if show_lines else None


        cnt_img = np.zeros(th.shape[:2], np.uint8)
        # cnt_img = biggest_contour(th) # todo: hull or not hull
        cnt = biggest_contour(th, with_convex=True)
        cv2.drawContours(cnt_img, [cnt], 0, 255, 5)

        lines = get_lines(cnt_img)
        try:
            matrix, size = get_matrix(lines)
        except IndexError:
            self.show(th, cnt_img, draw_lines(self.img, lines))
            return self.img

        cropped = cv2.warpPerspective(self.img, matrix, size)
        cropped = rotate_to_horizontal(cropped)

        if show_lines:
            concat = concatenate_images(th, show_cnt, draw_lines(self.img, lines))
            imgs_to_show.append(concat)
        if show_result:
            self.show(*imgs_to_show, cropped, histogram=show_histogram)

        return cropped

    def test(self):
        plt.hist(self.gray.ravel(),256,[0,256])
        plt.axvline(self.gray.mean(), color='k', linestyle='dashed', linewidth=1)
        plt.show()
        th = make_threshold(self.gray)

        self.show(th)
        # threshold = make_threshold(self.gray)
        # image_con = biggest_contour(threshold)

        # self.show(histogram, threshold, image_con)

    def show(self, *images, histogram=False):
        if histogram:
            self.show_gray_hist()

        fig=plt.figure(figsize=(25, 25))
        rows = len(images) + 1
        fig.add_subplot(rows, 1, 1)
        plt.imshow(self.img)

        for i, image in enumerate(images, 2):
            fig.add_subplot(rows, 1, i)
            plt.imshow(image, cmap='gray')
        plt.show()

    def show_gray_hist(self):
        plt.hist(self.gray.ravel(),256,[0,256])
        medium = medium_in_histogram(self.gray)
        plt.axvline(medium, color='k', linestyle='dashed', linewidth=1)
        plt.show()


def crop(image):
    return Cropper(image).crop()


if __name__ == "__main__":

    # img = img_read('rawChecks/cheMG_20201201_132952.jpg')
    # _x = Cropper(img).crop(show_result=Trueck/check/I, show_histogram=True, show_lines=True)

    path = '/cheque_parser/ch_photos/'

    for i in [5, 7, 8, 45, 26, 27, 28, 29, 42]:
        img = img_read(path + f'{i}.jpg')
        # cropped_img = Cropper(img).crop()
        plt.imshow(crop(img))
        plt.show()

