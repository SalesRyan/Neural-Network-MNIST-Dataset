import pygame
from pygame.locals import *
import numpy as np
from skimage.transform import resize

class neuron:
    def __init__(self,color,x,y,size):
        self.color = color
        self.x = x
        self.y = y
        self.size = size
    
    def coor(self):
        return [self.x,self.y]
    
    def my_size(self):
        return self.size

    def my_color(self):
        return self.color

class line:
    def __init__(self,color,x_y_i,x_y_f,size):
        self.color = color
        self.x_y_i = x_y_i
        self.x_y_f = x_y_f
        self.size = size

def scenario(din_x,din_y,n_input,n_hidden,n_output,w_ih,w_ho,image,pred,true,input_value):
    
    neuron_input = []
    neuron_hidden = []
    neuron_output = []

    line_in_for_hi = []
    line_hi_for_ou = []
    color = (0,0,0)
    x = 50
    y = 5
    for n in range(n_input):
        y+=35
        neuron_input.append(neuron((126,217,228),x,y,10))

    x = 350
    y = 5
    i = 0
    for n in range(n_hidden):
        y+=35
        neuron_hidden.append(neuron((172,172,172),x,y,10))
        for n_i in neuron_input:
            s = abs(int(w_ih[i]*3))
            if s < 1:
                s=-1
            else:
                color = (0,0,0)

            line_in_for_hi.append(line(color,neuron_hidden[n].coor(),n_i.coor(),1+s))
            i += 1

    x = 650
    y = 150
    i = 0
    for n in range(n_output):
        y+=35
        neuron_output.append(neuron((48,52,53),x,y,10))
        for n_o in neuron_hidden:
            s = abs(int(w_ih[i]*3))
            if s < 1:
                s=-1
            else:
                color = (0,0,0)

            line_hi_for_ou.append(line((0,0,0),neuron_output[n].coor(),n_o.coor(),1+s))
            i += 1
    

    pygame.init()
    screen = pygame.display.set_mode([din_x,din_y])

    pygame.display.set_caption("MLP Train")

    done = False
    clock = pygame.time.Clock()
 
    #while not done:
    

    #clock.tick(2)
    gameDisplay = pygame.display.set_mode((din_x,din_y))
    
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT: 
            done=True 

    screen.fill((255,255,255))
    img = pygame.surfarray.make_surface(resize(image.transpose(),(100,100)))
    gameDisplay.blit(img, (700,300))

    font_72 = pygame.font.SysFont("comicsansms", 72)
    font_30 = pygame.font.SysFont("comicsansms", 30)
    
    color = (0,0,0)
    if (pred==true):
        color = (0,255,0)
    else:
        color = (255,0,0)
        
    inf_true = font_30.render('True', True, (0,0,0))
    inf_pred = font_30.render('Predict', True, (0,0,0))
    pred_text = font_72.render(str(pred), True, color) 
    gameDisplay.blit(pred_text, (850,300)) 
    gameDisplay.blit(inf_pred,((820,250)))
    gameDisplay.blit(inf_true,((720,250)))
    
    for l in line_in_for_hi:

        pygame.draw.line(screen,l.color, l.x_y_i, l.x_y_f,l.size)

    for l in line_hi_for_ou:
        
        pygame.draw.line(screen,l.color, l.x_y_i, l.x_y_f, l.size)

    for neu,v in zip(neuron_input,input_value):

        pygame.draw.circle(screen, np.int_(neu.my_color()*abs(v)/10),neu.coor(), neu.my_size())#code danger, maybe color invalid.

    for neu in neuron_hidden:
        pygame.draw.circle(screen, neu.my_color(),neu.coor(), neu.my_size())

    for neu, i in zip(neuron_output,range(len(neuron_output))):
        if i == pred:
            my_color = (126,217,228)
        else:
            my_color = (48,52,53)
            
        pygame.draw.circle(screen, my_color,neu.coor(), neu.my_size())

    pygame.display.flip()


#pygame.quit()
