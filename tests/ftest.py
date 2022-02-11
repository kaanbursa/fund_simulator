import pytest

def func(x):
    return x

def test_method1():
    x = 5
    y = 10
    assert x == y

def test_method2():
    x = 10
    y = 20
    assert x + 10 == y