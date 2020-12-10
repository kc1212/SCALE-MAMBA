#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt

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


def read_stdin(total_lines=100):
    out = []
    for _ in range(total_lines):
        out.append(float(sys.stdin.readline()))
    return out


if __name__ == "__main__":
    n = 10
    xs = read_stdin()[:n]
    ys = read_stdin()[:n]
    start_m = 0
    start_c = 5
    m, c = gradient_descent_mse(xs, ys, m=start_m, c=start_c, alpha=0.0002, iters=10)

    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    ax.plot([0, max(xs)], [x*m + c for x in [0, max(xs)]], 'r', label='End')
    ax.plot([0, max(xs)], [x*start_m + start_c for x in [0, max(xs)]], 'g', label='Start')
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Linear regression demo')
    ax.legend()
    ax.grid()
    plt.show()

