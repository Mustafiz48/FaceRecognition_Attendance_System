from Register import Register
from Trainer import Trainer
from Tester import Tester
import tkinter as tk
from tkinter import *

win = Tk()
win.title('ATHS')
win.geometry("700x350")

radio = IntVar()
Label(text="ATHS Face-Recognition based Attendance System!\n\n", font=('Aerial 14')).pack()
Label(text="Select one of the following action you want to perform:", font=('Aerial 11')).pack()
label = Label(win)
label.pack()


def start():
    # Define radiobutton for each options
    r1 = Radiobutton(win, text="Register New Candidate", indicatoron=0, width=30, padx=40, pady=5, border=3,
                     variable=radio, value=1, command=selection)
    r1.pack(anchor=N)
    r2 = Radiobutton(win, text="Train AI", indicatoron=0, width=30, padx=40, pady=5, border=3,
                     variable=radio, value=2, command=selection)
    r2.pack(anchor=N)
    r3 = Radiobutton(win, text="Take Attendance", indicatoron=0, width=30, padx=40, pady=5, border=3,
                     variable=radio, value=3, command=selection)
    r3.pack(anchor=N)

    exit_button = Button(win, text="Exit", width=30, padx=40, pady=5, border=3, command=win.destroy,
                         bg='gray80', activebackground='red')
    exit_button.pack(pady=50)
    win.update()
    win.mainloop()


# Define a function to get the output for selected option
def selection():
    selected = radio.get()
    win.update()

    win.destroy()

    if selected == 1:
        print("\nStarting registration module...\n")
        register = Register()
        register.take_image()

    elif selected == 2:
        print("\nStarting model training...\n")
        trainer = Trainer()
        trainer.train_model()

    elif selected == 3:
        print("\nStarting Attendance taking module...\n")
        tester = Tester()
        tester.test()

    else:
        print("Something is not right. Please try again later")
        win.destroy()


if __name__ == '__main__':
    # register = Register()
    # register.take_image()
    # trainer = Trainer()
    # trainer.train_model()
    # tester = Tester()
    # tester.test()
    start()
