from img_processing import img_processing as imgp
import cv2

image_path = 'img/input/cc.jpg'
image = cv2.imread(image_path)

output_folder = 'img/output'

# ----------------- Select preprocess method -----------------
method = imgp.preprocess_image
# method = imgp.preprocess_image_adaptive

# ----------------- Detecting regions -----------------
# Configure scale_percent, ksize and blocksize for different results
regions = imgp.detect_regions(image, method, ksize=(30, 30), scale_percent=40)
# regions = imgp.detect_regions(image, method, blocksize=9, scale_percent=40)

# ----------------- Detecting words and chars -----------------

# Region ===> Words ===> Chars (Chars in words in regions)
# i = 0
# for region in regions:
#     words = imgp.detect_words(region, ksize=(8, 20))
#     for word in words:
#         chars = imgp.detect_chars(word, ksize=(1, 10), title=f"Chars {i}")
#         imgp.write_image(chars, output_folder)
#         i += 1

# Region ===> Chars (Chars in regions)
i = 0
for region in regions:
    chars = imgp.detect_chars(region, ksize=(1, 10), title=f"Chars {i}", show_result=False)
    imgp.write_image(chars, output_folder)
    i += 1

cv2.waitKey(0)
cv2.destroyAllWindows()
