from img_processing import img_processing as imgp
import cv2

image_path = 'img/input/ii.png'
image = cv2.imread(image_path)

# method = imgp.preprocess_image_adaptive
# regions = imgp.detect_regions(image, method, blocksize=9, scale_percent=40)

method = imgp.preprocess_image
regions = imgp.detect_regions(image, method, ksize=(30, 30), scale_percent=80)


# i = 0
# for region in regions:
#     words = imgp.detect_words(region, ksize=(8, 20))
#     for word in words:
#         chars = imgp.detect_chars(word, ksize=(1, 10), title=f"Chars {i}")
#         imgp.write_image(chars, 'img/output')
#         i += 1

i = 0
for region in regions:
    chars = imgp.detect_chars(region, ksize=(1, 10), title=f"Chars {i}")
    i += 1

cv2.waitKey(0)
cv2.destroyAllWindows()
