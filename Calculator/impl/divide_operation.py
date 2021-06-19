from Calculator.base.base_operation import BaseOperation


class DivideOperation(BaseOperation):

    def __init__(self, a, b):
        super(DivideOperation, self).__init__(a, b)

    def process(self):
        print("Performing Division of {} and {} ".format(self.a, self.b))
        result1 = self.a / self.b
        result2 = self.b / self.a
        return "Result1 = {}".format(result1), "Result2 = {}".format(result2)

