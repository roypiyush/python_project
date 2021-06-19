from Calculator.base.base_operation import BaseOperation


class MultiplyOperation(BaseOperation):

    def __init__(self, a, b):
        super(MultiplyOperation, self).__init__(a, b)

    def process(self):
        print("Performing Multiplication of {} and {} ".format(self.a, self.b))
        return self.a * self.b
