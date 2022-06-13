def cy_fib2(int n):
    a, b = 0.0, 1.0
    for i in range(n):
        a, b = a + b, a
    return a
