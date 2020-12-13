#!/usr/bin/env python3
import sys
import mpld3
import matplotlib.pyplot as plt


def dfdm(xs, zs):
    """
    Recall that
    f(m, c) = (1/n) * sum(y_i - (m * x_i + c))^2.
    We differentiate f(m, c) with respect to m to obtain
    dfdm = (-2/n)*sum(x_i*(y_i - (m*x_i + c))).
    We optimize the above using
    z_i = y_i - (m*x_i + c) to arrive at
    dfdm = (-2/n)*sum(x_i*z_i).
    """
    return (-2/n)*sum([x*z for x, z in zip(xs, zs)])


def dfdc(zs):
    """
    We follow the exact same process as dfdm,
    firstly we differentiate to get
    dfdc = (-2/n)*sum(y_i - (m*x_i + c)),
    then optimize using z_i, so
    dfdc = (-2/n)*sum(z_i).
    """
    return (-2/n)*sum(zs)


def gradient_descent_mse(xs, ys, m, c, alpha, iters):
    """
    The goal of linear regression is to build a linear model y = m*x + c
    using a set of samples/points (x, y).

    This function implements the
    gradient descent algorithm for the mean squared error
    f(m, c) = (1/n) * sum(y_i - (m * x_i + c))^2.

    Below you'll find the input arguments.
    Note that when implementing this function in MPC,
    xs and ys are secret but the rest are shared.

    xs: the x-coordinates (party 0)
    ys: the y-coordinates (party 1)
    m: gradient
    c: constant
    alpha: learning rate
    iters: number of iterations for gradient descent
    """
    assert len(xs) == len(ys)
    n = float(len(xs))

    for _ in range(iters):
        zs = [y - (m*x + c) for x, y in zip(xs, ys)]
        tmp_m = m - alpha*dfdm(xs, zs)
        tmp_c = c - alpha*dfdc(zs)
        m = tmp_m
        c = tmp_c

    return (m, c)


def read_stdin(total_lines):
    out = []
    for _ in range(total_lines):
        out.append(float(sys.stdin.readline()))
    return out


if __name__ == "__main__":
    n = 10
    xs = read_stdin(n)
    ys = read_stdin(n)
    start_m = 0
    start_c = 5
    m, c = gradient_descent_mse(xs, ys, m=start_m, c=start_c, alpha=0.0002, iters=10)
    print("m: {}, c: {}".format(m, c))

    # Plotting
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    ax.plot([0, max(xs)], [x*m + c for x in [0, max(xs)]], 'r', label='Clear')
    ax.plot([0, max(xs)], [x*start_m + start_c for x in [0, max(xs)]], 'g', label='Start')
    ax.set_ylim(0)
    ax.set_xlim(0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Linear regression demo')
    ax.legend()
    ax.grid()
    mpld3.show()

