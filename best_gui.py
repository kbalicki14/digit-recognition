import tkinter as tk
import numpy as np
import cv2
from keras.models import load_model


from PIL import ImageTk, Image, ImageDraw

model_load = load_model('model_mnist.h5')


def event_function(event):
    x = event.x
    y = event.y
    x1 = x - 8
    y1 = y - 8
    x2 = x + 8
    y2 = y + 8

    canvas.create_oval((x1, y1, x2, y2), fill='black')
    img_draw.ellipse((x1, y1, x2, y2), fill='white')


def save():
    global count

    img_array = np.array(img)
    img_array = cv2.resize(img_array, (28, 28))

    cv2.imwrite(str(count) + '.jpg', img_array)
    count = count + 1


def clear():
    global img, img_draw

    canvas.delete('all')
    img = Image.new('RGB', (500, 500), (0, 0, 0))
    img_draw = ImageDraw.Draw(img)

    label_status.config(text='PREDICTED DIGIT: NONE')


def predict():
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (28, 28))

    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)
    result = model_load.predict([img_array])[0]
    digit, acc = np.argmax(result), max(result)

    # print(acc)

    label_status.config(text='Guess it is: ' + str(digit) + ' with prob '+ str(int(acc * 100)) +' %')


count = 0

win = tk.Tk()

win.title("Digit recognition")

canvas = tk.Canvas(win, width=500, height=500, bg='white')
canvas.grid(row=0, column=0, columnspan=4)

button_save = tk.Button(win, text='SAVE', bg='green', fg='white', font='Helvetica 20 bold', command=save)
button_save.grid(row=1, column=0)

button_predict = tk.Button(win, text='PREDICT', bg='blue', fg='white', font='Helvetica 20 bold', command=predict)
button_predict.grid(row=1, column=1)

button_clear = tk.Button(win, text='CLEAR', bg='yellow', fg='black', font='Helvetica 20 bold', command=clear)
button_clear.grid(row=1, column=2)

button_exit = tk.Button(win, text='EXIT', bg='red', fg='white', font='Helvetica 20 bold', command=win.destroy)
button_exit.grid(row=1, column=3)

label_status = tk.Label(win, text='PREDICTED DIGIT: NONE', bg='white', font='Helvetica 24')
label_status.grid(row=2, column=0, columnspan=4)

canvas.bind('<B1-Motion>', event_function)


img = Image.new('RGB', (500, 500), (0, 0, 0))
img_draw = ImageDraw.Draw(img)

win.mainloop()
