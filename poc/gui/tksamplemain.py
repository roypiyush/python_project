from tkinter import *


class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.number = 0
        self.widgets = []
        self.clone_button = None
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        self.clone_button = Button (self, text='Clone', command=self.clone)
        self.clone_button.grid()

    def clone(self):
        widget = Label(self, text='label #%s' % self.number)
        widget.grid()
        self.widgets.append(widget)
        self.number += 1


if __name__ == "__main__":
    app = Application()
    app.master.title("Sample application")
    app.mainloop()
