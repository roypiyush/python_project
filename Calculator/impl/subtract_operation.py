from Calculator.base.base_operation import BaseOperation


class SubtractOperation(BaseOperation):

    def __init__(self, a, b):
        super(SubtractOperation, self).__init__(a, b)

    def process(self):
        print("Performing Subtraction of {} and {} ".format(self.a, self.b))
        return "Result1 = {}".format(self.a - self.b), "Result2 = {}".format(self.b - self.a)
