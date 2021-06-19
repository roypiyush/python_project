import pytest


@pytest.fixture(scope='session')
def setup(request):
    print('setup()')

    def teardown():
        print('teardown()')
    request.addfinalizer(teardown)


def test_1_that_needs_resource_a(setup):
    print('test_1_that_needs_resource_a()')


def test_2_that_does_not():
    print('test_2_that_does_not()')


def test_3_that_does(setup):
    print('test_3_that_does()')

