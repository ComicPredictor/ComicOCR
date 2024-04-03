from comicsocr import  api
import cv2

img = cv2.imread("comic_data\\test\\opm.jpg", cv2.IMREAD_COLOR) 
h_img, w_img, _ = img.shape

data=api.read_from_file("comic_data\\test\\opm.jpg")
texes=[]
for i, bbox in data:
    s=''
    for j in i.split('\n'):
        b=0
        if j:
            char, x, y, w, h, _ = j.split(" ")
            x, y, w, h = int(x), int(y), int(w), int(h)
            if char and char.lower() in "abcdefghijklmnopqrstuv,.?!" and (x!= 0 and y != 0):
                if b==0:
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    b=1
                s+=char
    if s:
        texes.append((s, bbox))
    
print(texes)
cv2.imshow("iomg", img)
cv2.waitKey(0)            