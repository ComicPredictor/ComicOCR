from comicsocr import  api
import logging
import cv2
def textnpos(image_path:str, show:bool=False, log:bool=False):
    if not log:
        logging.getLogger('comicsocr').propagate=False
    img = cv2.imread(image_path, cv2.IMREAD_COLOR) 
    #h_img, w_img, _ = img.shape

    data=api.read_from_file(image_path)
    texes=[]
    for i, bbox in data:
        s=''
        for j in i.split('\n'):
            b=0
            if j:
                char, x, y, w, h, _ = j.split(" ")
                x, y, w, h = int(x), int(y), int(w), int(h)
                if char and char.lower() in "abcdefghijklmnopqrstuv,.?!" and x!= 0 and y != 0:
                    if b==0 and show:
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    b=1
                    s+=char
        if s:
            texes.append((s, bbox))
        
    if show:
        cv2.imshow("iomg", img)
        cv2.waitKey(0)  
    return texes          