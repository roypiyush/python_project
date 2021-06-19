from Calculator.impl import add_operation
from Calculator.impl import subtract_operation
from Calculator.impl import multiply_operation
from Calculator.impl import divide_operation
import re
import sys
import signal

NUMBER_REGEX = re.compile("^([1-4])$")


def signal_handler(*args):
    print(''.format(args))
    print('Exiting program !!!')
    # for p in jobs:
    #     p.terminate()
    sys.exit(0)


def start_calculator(operation_to_perform, a_num, b_num):

    if operation_to_perform == "1":
        add = add_operation.AddOperation(a_num, b_num)
        print(add.process())
    elif operation_to_perform == "2":
        subtract = subtract_operation.SubtractOperation(a_num, b_num)
        print(subtract.process())
    elif operation_to_perform == "3":
        multiply = multiply_operation.MultiplyOperation(a_num, b_num)
        print(multiply.process())
    elif operation_to_perform == "4":
        divide = divide_operation.DivideOperation(a_num, b_num)
        print(divide.process())
    else:
        print("Invalid Input [Operation = {}]. Not yet Implemented".format(operation_to_perform))


def enter_number(prompt_string):

    while True:
        try:
            num = int(input(prompt_string))
            return num
        except ValueError:
            print("Could not convert data to an integer. Please enter a number.")


def verify_operation_input(regex_match, prompt_string):
    value_string = input(prompt_string)
    while regex_match.match(value_string) is None:
        value_string = str(input(prompt_string))

    return value_string


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    print("Enter 1 to add "
          "\nEnter 2 to subtract "
          "\nEnter 3 to multiply "
          "\nEnter 4 to divide "
          "\nCtrl C to quit\n")
    operation = verify_operation_input(NUMBER_REGEX, "Enter number between 1-4 ")
    a = enter_number("Enter number A = ")
    b = enter_number("Enter number B = ")
    start_calculator(operation, a, b)
