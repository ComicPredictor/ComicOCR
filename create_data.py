from comicsocr import  api
import cv2

img = cv2.imread("comic_data\\test\\opm.jpg", cv2.IMREAD_COLOR) 
h_img, w_img, _ = img.shape

data=api.read_from_file("comic_data\\test\\opm.jpg")
for i in data[1]:
    for j in i.split('\n'):
        if j:
            char, x, y, w, h, _ = j.split(" ")
            x, y, w, h = int(x), int(y), int(w), int(h)
            if x != "0" or y != "0":
                cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)

cv2.imshow("iomg", img)
cv2.waitKey(0)            