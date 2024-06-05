import usecomicsocr
import usellama


def create_dialogue_options(imgpath, txtpath):
    options=[]
    ocrd, _=usecomicsocr.textnpos(imgpath)
    
    for s, _ in ocrd:
        ns=usellama.decode(s)
        with open(txtpath, 'a') as o:
            o.write(ns+' \n')
        options.append(ns)
    return options
if __name__=='__main__':
    print(create_dialogue_options("comic_data\\test\\image.png", "test.txt"))
        
        
        



# import pygame as pg
# import pygame_textinput as pgti

# pg.init()
# textinput = pgti.TextInputVisualizer()
# #設定視窗
# width, height = 800, 800                     
# screen = pg.display.set_mode((width, height))   
# pg.display.set_caption("Sean's game")         
# #建立畫布bg
# bg = pg.Surface(screen.get_size())
# bg = bg.convert()
# bg.fill((255,255,255))

# image = pg.image.load("comic_data/test/image.png")
# image = pg.transform.scale(image, (700,700))
# image.convert()
# bg.blit(image, (20,10))


# #要在這兩條程式碼上面喔!!
# screen.blit(bg, (0,0))
# pg.display.update()


# #關閉程式的程式碼
# running = True
# while running:
#     for event in pg.event.get():
#         if event.type == pg.QUIT:
#             running = False
# pg.quit()   