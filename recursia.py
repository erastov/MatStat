import timeit
def factorial(n):
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res

def factorialrec(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

data = 600
print(timeit.timeit("factorial(data)", setup="from __main__ import factorial, data", number=1))

