import cv2

image_path = "img/input/rsz.png"

image = cv2.imread(image_path)
dim = (32, 32)
image2 = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imwrite("img/output/rsz.png", image2)
cv2.imshow("Resized image", image2)

cv2.waitKey(0)
cv2.destroyAllWindows()