# loading of test image and annotation - OpenCV and JSON
import os
import cv2
import json

img = cv2.imread(os.path.join('../data', 'train', 'images', '2a52abe3-fd5f-11ed-806b-b42e991176dc.jpg'))

with open(os.path.join('../data', 'train', 'labels', '2a52abe3-fd5f-11ed-806b-b42e991176dc.json'), 'r') as f:
    label = json.load(f)

cv2.imshow("test_image", img)
print(label)
cv2.waitKey(0)

cv2.destroyAllWindows()
