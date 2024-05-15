from tkinter import *
import numpy as np

height = 500
width = 400

class barGraph():
    height: int
    y0: int
    x0: int
    x1: int
    def __init__(self, height = 250, left_or_right = False):
        self.height = height
        self.y0 = 500
        #left = false, right = true
        if left_or_right:
            self.x0 = 210
            self.x1 = 390
        else:
            self.x0 = 10
            self.x1 = 190
    def update_height(self, height):
        self.height = 500 - height
    
    def increase_height(self, increase):
        self.height = self.height + increase
root = Tk()
root.geometry(str(width)+"x"+str(height))
root.title("Moving Bar Graph")

mainCanvas = Canvas(root, width = (width), height = (height) + 100, bg= 'white')
mainCanvas.pack()

left_bar = barGraph(100, False)
right_bar = barGraph(200, True)

def draw_rec(rec):
    mainCanvas.create_rectangle(rec.x0, rec.y0, rec.x1, height/2)

def update_rec(rec):
    mainCanvas.create_rectangle(rec.x0, rec.y0, rec.x1 ,0,fill = "black", outline= "white")
    mainCanvas.create_rectangle(rec.x0, rec.y0, rec.x1, rec.height)

def leftClick(event):
    mainCanvas.create_rectangle(0,0,400,500, fill = 'black')
    left_bar.update_height(250)
    right_bar.update_height(250)
    draw_rec(right_bar)
    draw_rec(left_bar)

def rightClick(event):
    left_bar.increase_height(-15)
    right_bar.increase_height(15)
    update_rec(right_bar)
    update_rec(left_bar)

root.bind("<Button-2>", rightClick)
root.bind("<Button-3>", leftClick)

root.mainloop()


    