import os
import random
import string
import time

import cv2
import numpy as np


def resize_image(image, scale_percent=100):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def preprocess_image(image, ksize):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(255 - gray_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh_image, rect_kernel, iterations=1)
    return dilation


def preprocess_image_adaptive(image, blockSize, C=2):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    thresh_adaptive = cv2.adaptiveThreshold(255 - denoised_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                            blockSize, C)
    # cv2.imshow("Adaptive Threshold", thresh_adaptive)

    return 255 - thresh_adaptive


def find_contours(thresh_image):
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours(contours, component):
    filtered_contours = []
    areas = [cv2.contourArea(cnt) for cnt in contours]
    median_area = np.median(areas)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter based on aspect ratio, area, and other desired criteria
        if component == "region":
            if area > 300 and 0.5 < w / h:
                filtered_contours.append(cnt)
        elif component == "word":
            if area > 300 and 0.5 < w / h < 10:
                filtered_contours.append(cnt)
        elif component == "char":
            if median_area / 5 < area < median_area * 2 and 0.1 <= w / h <= 2.5:
                filtered_contours.append(cnt)

    return filtered_contours


# def filter_region_contours(contours):
#     filtered_contours = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         x, y, w, h = cv2.boundingRect(cnt)
#         # Filter based on aspect ratio, area, and other desired criteria
#         if area > 300 and 0.5 < w / h:
#             filtered_contours.append(cnt)
#
#     return filtered_contours
#
#
# def filter_word_contours(contours):
#     filtered_contours = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         x, y, w, h = cv2.boundingRect(cnt)
#         # Filter based on aspect ratio, area, and other desired criteria
#         if area > 300 and 0.5 < w / h < 10:
#             filtered_contours.append(cnt)
#
#     return filtered_contours
#
#
# def filter_char_contours(contours):
#     # Calculate the area for each contour and store in a list
#     areas = [cv2.contourArea(cnt) for cnt in contours]
#
#     # Calculate the median of the areas
#     median_area = np.median(areas)
#
#     # Filter contours
#     filtered_contours = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         x, y, w, h = cv2.boundingRect(cnt)
#         # Filter based on aspect ratio, area, and other desired criteria
#         if median_area / 5 < area < median_area * 2 and 0.1 <= w / h <= 2.5:
#             filtered_contours.append(cnt)
#
#     return filtered_contours


def detect_regions(image, preprocess_method=preprocess_image, ksize=(25, 30), blocksize=9, scale_percent=60):
    resized_image = resize_image(image, scale_percent=scale_percent)

    global preprocessed_image
    if preprocess_method == preprocess_image_adaptive:
        preprocessed_image = preprocess_method(resized_image, blocksize)
    elif preprocess_method == preprocess_image:
        preprocessed_image = preprocess_method(resized_image, ksize)

    contours = find_contours(preprocessed_image)
    region_contours = filter_contours(contours, "region")

    regions = []
    image2 = resized_image.copy()
    for i, contour in enumerate(region_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 1)
        regions.append(resized_image[y:y + h, x:x + w])
    cv2.imshow("Regions", image2)

    return regions


def detect_words(region, preprocess_method=preprocess_image, ksize=(5, 8)):
    preprocessed_image = preprocess_method(region, ksize)

    contours = find_contours(preprocessed_image)
    word_contours = filter_contours(contours, "word")
    print(len(contours), len(word_contours))

    words = []
    region2 = region.copy()
    for i, contour in enumerate(word_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(region2, (x, y), (x + w, y + h), (0, 255, 0), 1)
        words.append(region[y:y + h, x:x + w])
    cv2.imshow("Words", region2)

    return words


def detect_chars(word, preprocess_method=preprocess_image, ksize=(1, 8), title="Chars"):
    preprocessed_image = preprocess_method(word, ksize)

    contours = find_contours(preprocessed_image)
    char_contours = filter_contours(contours, "char")
    print(len(contours), len(char_contours))

    chars = []
    word2 = word.copy()
    for i, contour in enumerate(char_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(word2, (x, y), (x + w, y + h), (0, 255, 0), 1)
        chars.append(word[y:y + h, x:x + w])
    cv2.imshow(title, word2)

    return chars


def write_image(images, output_folder):
    for image in images:
        timestamp = int(time.time())
        random_string = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))

        filename = f"{timestamp}_{random_string}.png"
        filepath = os.path.join(output_folder, filename)

        # Write the character image to the file
        cv2.imwrite(filepath, image)
