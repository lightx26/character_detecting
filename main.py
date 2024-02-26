from img_processing import img_processing as imgp
import cv2

image_path = 'img/input/line.jpg'
image = cv2.imread(image_path)

output_folder = 'img/output'

# ----------------- Select preprocess method -----------------
# method = imgp.preprocess_image
method = imgp.preprocess_image_adaptive

# ----------------- Detecting regions -----------------
# Configure scale_percent, ksize and blocksize for different results
# regions = imgp.detect_regions(image, method, ksize=(25, 30))
regions = imgp.detect_regions(image, method, blocksize=9)


# ----------------- Detecting words and chars -----------------

# Region ===> Words ===> Chars (Chars in words in regions)
# i = 0
# for region in regions:
#     words = imgp.detect_words(region, ksize=(6, 10))
#     for word in words:
#         chars = imgp.detect_chars(word, ksize=(i, 9), title=f"Chars {i}", show_result=False)
#         # imgp.write_image(chars, output_folder)
#         i += i

# Region ===> Chars (Chars in regions)
i = 0
for region in regions:
    chars = imgp.detect_chars(region, ksize=(1, 5), title=f"Chars {i}", show_result=False)
    imgp.write_image(chars, output_folder, mk_folder=True)
    i += 1

# Press enter to destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
