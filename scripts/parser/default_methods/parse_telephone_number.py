#%%
from cv2 import cv2 
import matplotlib.pyplot as plt
import pytesseract
import os
#%%
def combine_in_order(items, max_step=3):
    for i in range(len(items)):
        slice_ = []
        for _i in range(i, min(len(items), i + max_step), 1):
            slice_.append(items[_i])
            yield slice_

def is_phone_number(text):
    score = 0
    delta = abs(sum(i.isdigit() for i in text) - 10)
    score += 0.6 / (delta / 3 + 1)
    score += 0.4 * ('-' in text)
    return score

def phone_score(text_line):
    score = 80 * max((is_phone_number(''.join(line)) for line in combine_in_order(text_line)), default=0)
    num_of_digits = sum(i.isdigit() for i in ''.join(text_line)) 
    if num_of_digits > 6:
        score += 20 * ('ל' in ''.join(text_line))
        
        score += 20 * ('ט' in ''.join(text_line))

    score += 40 * (text_line[-1][0] == '0')
    return score

def _find_line_of_numbers(data):
    lines = []
    for i, text in enumerate(data['text']):
        if text:
            if i and data['text'][i - 1]:
                lines[-1].append(i)
            else:
                lines.append([i])

    if not lines:
        return None
    if len(lines) > 1:
        scores = []
        for line in lines:
            text_line = [data['text'][i] for i in line]
            score = phone_score(text_line)
            if score < 30:
                score = -1
            scores.append(score)
        if max(scores) < 0:
            return None
        number_indxs = lines[scores.index(max(scores))]
    else:
        number_indxs = lines[0]
    
    return number_indxs

def get_line_of_numbers(img, _threshholded=False):
    data = pytesseract.image_to_data(img, output_type='dict', config='--psm 6', lang='heb')
    number_indxs = _find_line_of_numbers(data)
    if number_indxs is None:
        if _threshholded:
            return None, None
        _, _th = cv2.threshold(img, img.mean() * 0.9, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        _th = 255 - _th
        return get_line_of_numbers(_th, True)
    return number_indxs, data

def get_rectangle(data, word_indxs):
    left = min(10, data['left'][word_indxs[0]])
    right = max(230, data['left'][word_indxs[-1]] + data['width'][word_indxs[-1]])
    top = sum(data['top'][i] for i in word_indxs) // len(word_indxs)
    bottom = sum(data['height'][i] for i in word_indxs) // len(word_indxs) + top

    return left, top, right, bottom

def expand_rectangle(left, top, right, bottom, max_shape, increase=1.5, max_width=None):
    delta_width = int((bottom - top) * (increase - 1))
    top = max(0, top - delta_width)
    left = max(0, left - 5)
    right = min(max_shape[1], right + 5)
    bottom = min(max_shape[0], bottom + delta_width)
    if max_width is not None:
        med = (top + bottom) / 2
        top = int(max(top, med - max_width / 2))
        bottom = int(min(bottom, med + max_width / 2))
        
    return left, top, right, bottom

def crop_telephon_numbers(telephon_area):
    draw_and_show_boxes(telephon_area) #remove
    number_indxs, data = get_line_of_numbers(telephon_area)
    if not number_indxs:
        return None
    
    rect = get_rectangle(data, number_indxs)
    left, top, right, bottom = expand_rectangle(*rect, max_shape=telephon_area.shape, max_width=25)
    # plt.imshow(telephon_area[top :bottom , left :right])
    # plt.show()
    return telephon_area[top :bottom , left :right].copy()

def draw_and_show_boxes(img, lang='heb', config=''):
    boxes = pytesseract.image_to_boxes(img, lang=lang, config=config)
    _img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h = img.shape[0]
    for b in boxes.splitlines():
        b = b.split(' ')
        color = (255, 0, 0) if b[0].isdigit() and b[5] == '0' else (0, 0, 255)
        _img = cv2.rectangle(_img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), color, 1)
    # plt.imshow(_img)
    # plt.show()

def find_phone_numbers(number_image, _threshholded=False):
    data = pytesseract.image_to_data(number_image, lang='eng', config='--psm 7 -c tessedit_char_whitelist="0123456789 -"', output_type='dict')
    # print(data[''])
    text = ''.join(data['text'])
    numbers = ''.join(t for t in text if t.isdigit())
    if 8 < len(numbers) < 11:
        # conf = [c for c in data['conf'] if c > 0]
            
        numbers = '0' + numbers[1:]
        return [numbers]#, [sum(conf) / len(conf)]
    
    list_numbers = [''.join(i for i in t if i.isdigit()) for t in data['text']]
    conf = [c for c, n in zip(data['conf'], list_numbers) if n]
    # list_numbers = [n for n in ]
    _numbers_list = []
    for num in list_numbers:
        if 7 < len(num) < 11:
            if len(num) == 8:
                num = '0' + num
            elif num[0] != '0':
                num = '0' + num[1:]
            _numbers_list.append(num)
    if not _threshholded and not _numbers_list:
        _, number_image = cv2.threshold(number_image, 120, 255, cv2.THRESH_BINARY)
        return find_phone_numbers(number_image,_threshholded=True)
    return _numbers_list or None

def parse_telephone_numbers(cropped_gray):
    img = cv2.resize(cropped_gray, (900, 400))
    number_area = img[50:130, 650:-10]
    number_image = crop_telephon_numbers(number_area)
    # plt.imshow(number_image)
    # plt.show()
    if number_image is None:
        return {'numbers': None}
    return {'numbers': find_phone_numbers(number_image)}
    

# #%%
# names = os.listdir('cropped/')
# names.sort(key=lambda x: int(x.split('.')[0]))
# for name in names:
#     img = cv2.imread('cropped/' + name, 0)

#     print(parse_telephone_numbers(img))
        
        









# # %%
# name = 19
# img = cv2.imread('cropped/' + str(name) + '.jpg', 0)

# img = cv2.resize(img, (900, 400))
# number_area = img[50:130, 650:-10]
# # number_area = img[80:100, 770:-10]
# number_image = crop_telephon_numbers(number_area, 1)
# if number_image is not None:
#     plt.imshow(number_area)
#     plt.show()
#     data = pytesseract.image_to_data(number_image, lang='Hebrew', config='--psm 7 -c tessedit_char_whitelist="0123456789 -"', output_type='dict')


#     print(name, data['text'])
# else:
#     print(name)
# # %%

# blured = cv2.GaussianBlur(number_area, (3, 3), 0.5)
# _, th = cv2.threshold(number_area, 90, 255, cv2.THRESH_BINARY)
# plt.imshow(th)
# #%%
# # number_area_ = number_area[10:45, 105:215]
# draw_and_show_boxes(number_area, 'heb', '')

# # %%
# # config = '--psm 7 -c tessedit_char_whitelist="0123456789 -"'
# data = pytesseract.image_to_data(number_area, lang='heb', config='', output_type='dict')
# data['text']

# %%
