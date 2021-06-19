from Calculator.base.base_operation import BaseOperation


class AddOperation(BaseOperation):

    def __init(self, a, b):
        super(AddOperation, self).__init(a, b)

    def process(self):
        print("Performing Addition of {} and {} ".format(self.a, self.b))
        return self.a + self.b
