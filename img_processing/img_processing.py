import os
import random
import string
import time

import cv2
import numpy as np
import pytesseract


def resize_image(image, scale_percent=100, new_width=1000):
    if scale_percent == 100:
        scale_percent = new_width / image.shape[1] * 100

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def preprocess_image(image, ksize, component):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(255 - gray_image, 127, 255, cv2.THRESH_OTSU)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

    #Applying erosion on the threshold image


    # Applying dilation on the threshold image

    if component == "char":
        dst = cv2.morphologyEx(thresh_image, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
        dst = cv2.dilate(dst, rect_kernel, iterations=1)
    else:
        dst = cv2.dilate(thresh_image, rect_kernel, iterations=1)

    # # Display the image
    # random_string = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
    # cv2.imshow(random_string, dst)
    return dst


def thinner(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # Create an empty output image to hold values
    thin = np.zeros(gray.shape, dtype='uint8')
    img1 = gray.copy()

    # Loop until erosion leads to an empty set
    while cv2.countNonZero(img1) != 0:
        # Erosion
        erode = cv2.erode(img1, kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset, thin)
        # Set the eroded image for next iteration
        img1 = erode.copy()
    return thin


def preprocess_image_adaptive(image, blockSize, C=2):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    thresh_adaptive = cv2.adaptiveThreshold(255 - denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            blockSize, C)
    # cv2.imshow("Adaptive Threshold", thresh_adaptive)

    # Display the image
    # random_string = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
    # cv2.imshow(random_string, thresh_adaptive)

    return 255 - thresh_adaptive


def find_contours(thresh_image):
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours(contours, component):
    filtered_contours = []

    mean_area = np.mean([cv2.contourArea(cnt) for cnt in contours])
    # mean_width = np.mean([cv2.boundingRect(cnt)[2] for cnt in contours])
    mean_height = np.mean([cv2.boundingRect(cnt)[3] for cnt in contours if cv2.boundingRect(cnt)[3] > 30])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter based on aspect ratio, area, and other desired criteria
        if component == "region":
            if mean_height * 0.75 < h < mean_height * 1.5:
                filtered_contours.append(cnt)
        elif component == "word":
            if area > 300 and 0.5 < w / h < 10:
                filtered_contours.append(cnt)
        elif component == "char":
            if mean_area / 5 < area < mean_area * 2.5 and 0.1 <= w / h <= 2.5:
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
#         if median_area / 5 < area < median_area * 2 and 0.i <= w / h <= 2.5:
#             filtered_contours.append(cnt)
#
#     return filtered_contours


def detect_regions(image, preprocess_method=preprocess_image, ksize=(25, 30), blocksize=9, scale_percent=60):
    resized_image = resize_image(image, scale_percent=scale_percent)

    global preprocessed_image
    if preprocess_method == preprocess_image_adaptive:
        preprocessed_image = preprocess_method(resized_image, blocksize)
    elif preprocess_method == preprocess_image:
        preprocessed_image = preprocess_method(resized_image, ksize, "region")

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


def detect_words(parent_img, preprocess_method=preprocess_image, ksize=(5, 8)):
    preprocessed_image = preprocess_method(parent_img, ksize, "word")

    contours = find_contours(preprocessed_image)
    word_contours = filter_contours(contours, "word")
    # sorted_contours = sort_contours(word_contours, method="left-to-right")

    words = []
    parent_img2 = parent_img.copy()
    for i, contour in enumerate(word_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(parent_img2, (x, y), (x + w, y + h), (0, 255, 0), 1)
        words.append(parent_img[y:y + h, x:x + w])
    cv2.imshow("Words", parent_img2)

    return words


def detect_chars(parent_img, preprocess_method=preprocess_image, ksize=(1, 8), title="Chars", show_result=False):
    preprocessed_image = preprocess_method(parent_img, ksize, "char")

    contours = find_contours(preprocessed_image)
    char_contours = filter_contours(contours, "char")
    wid = np.empty(len(char_contours))

    chars = []
    parent_img2 = parent_img.copy()
    for i, contour in enumerate(char_contours):
        x, y, w, h = cv2.boundingRect(contour)
        # np.insert(wid, 0, (w, h))
        cv2.rectangle(parent_img2, (x, y), (x + w, y + h), (0, 255, 0), 1)
        char_img = parent_img[y:y + h, x:x + w]
        chars.append(char_img)

    if show_result:
        cv2.imshow(title, parent_img2)

    # np.sort(wid)
    # print(wid)

    return chars


def __convert_to_normal_char(char):
    normal_char = "abcdefghijklmnopqrstuvwxyz"
    if char in normal_char:
        return char

    # Define a dictionary mapping special characters to their ASCII equivalents
    special_char_map = {
        "a": ["à", "á", "ả", "ã", "ạ", "â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "ă", "ằ", "ắ", "ẳ", "ẵ", "ặ"],
        "e": ["è", "é", "ẻ", "ẽ", "ẹ", "ê", "ề", "ế", "ể", "ễ", "ệ"],
        "i": ["ì", "í", "ỉ", "ĩ", "ị"],
        "o": ["ò", "ó", "ỏ", "õ", "ọ", "ô", "ồ", "ố", "ổ", "ỗ", "ộ", "ơ", "ờ", "ớ", "ở", "ỡ", "ợ"],
        "u": ["ù", "ú", "ủ", "ũ", "ụ", "ư", "ừ", "ứ", "ử", "ữ", "ự"],
        "y": ["ỳ", "ý", "ỷ", "ỹ", "ỵ"],
        "d": ["đ"],
    }

    for key in special_char_map:
        if char in special_char_map[key]:
            return key

    # If not found, return the original character
    return char


def write_image(images, output_folder, mk_folder=False):
    for image in images:

        timestamp = int(time.time())
        random_string = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
        filename = f"{timestamp}_{random_string}.png"

        # Use Tesseract to recognize the character
        if (mk_folder):
            text = pytesseract.image_to_string(image, config='-l vie --psm 10 -c '
                                                             'tessedit_char_blacklist=!@#$%^&*<>?|/[]{}<>').strip().lower()
            text = ''.join([__convert_to_normal_char(char) for char in text])
            # Create the folder if it doesn't exist
            text_folder = os.path.join(output_folder, text)
            os.makedirs(text_folder, exist_ok=True)
            filepath = os.path.join(output_folder, text, filename)

        else:
            filepath = os.path.join(output_folder, filename)

        # Write the character image to the file
        cv2.imwrite(filepath, image)


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)
