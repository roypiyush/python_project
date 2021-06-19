def func(x):
    return x + 1


def test_answer():
    print 'Starting pytest'
    a = 4
    assert func(3) == a
