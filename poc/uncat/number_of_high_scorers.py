

if __name__ == '__main__':

    students_with_total_marks = {}
    students = set()
    with open('input.txt', 'r') as input_file:
        data = ""
        for data in input_file:
            sid, subject, marks = map(lambda x: x.strip(), data.split(','))
            students_with_total_marks[sid] = students_with_total_marks.get(sid, 0) + int(marks)
            if students_with_total_marks[sid] >= 100:
                students.add(sid)

    with open('output.txt', 'w') as output_file:
        file.write(str(len(students)))
