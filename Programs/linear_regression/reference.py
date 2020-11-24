#!/usr/bin/env python3
import sys


def gradient_descent_mse(xs, ys, m = 0.0, c = 5.0, alpha=0.00001, iters=1000):
    """
    Gradient descent algorithm for the mean squared error
    f(m, c) = (1/n) * sum(yi - (m * xi + c))^2.
    """
    xys = list(zip(xs, ys))
    n = float(len(xys))

    inner = lambda m, c : [(y - (m*x + c)) for x, y in xys]
    dfdc = lambda ins : (-2/n)*sum(ins)
    dfdm = lambda ins : (-2/n)*sum([t*x for t, (x, _) in zip(ins, xys)])

    for _ in range(iters):
        tmp = inner(m, c)
        tmp_m = m - alpha*dfdm(tmp)
        tmp_c = c - alpha*dfdc(tmp)
        m = tmp_m
        c = tmp_c

    return (m, c)


def read_stdin(n=10):
    out = []
    for _ in range(n):
        out.append(float(sys.stdin.readline()))
    return out


if __name__ == "__main__":
    xs = read_stdin()
    ys = read_stdin()
    print(gradient_descent_mse(xs, ys, alpha=0.0002, iters=10))

