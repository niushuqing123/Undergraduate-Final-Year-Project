import cv2
import numpy
# numpy.set_printoptions(threshold=numpy.inf)

img = cv2.imread("data_train/mask/000500.png")

m=img.max()

img=img*127
cv2.imshow("Test Image", img)
cv2.waitKey(0)

#检验批量制作的mask是否正确