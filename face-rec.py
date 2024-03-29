import tensorflow as tf
import cv2
import numpy as np
print(tf.config.list_physical_devices())
model=tf.saved_model.load('saved_model')
a=cv2.imread("comic_data/test/show-your-cool-best-favourite-one-piece-manga-panel-v0-giqm40q1voy91-ezgif.com-webp-to-jpg-converter.jpg")
def yolobbox2bbox(x,y,w,h):
    H, W, _=a.shape
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return (int(H/2+H*x1/2), int(W/2+W*y1/2)), (int(H/2+H*x2/2), int(W/2+W*y2/2))
yolobboxes=model(np.expand_dims(a, axis=0))['raw_detection_boxes']

for i in yolobboxes[0]:
    a=cv2.rectangle(a, *yolobbox2bbox(*i), (0, 255, 0), 3)
cv2.imshow('d', a)
cv2.waitKey(0)
cv2.destroyAllWindows()