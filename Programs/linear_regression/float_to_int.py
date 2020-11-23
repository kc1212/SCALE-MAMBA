#!/usr/bin/env python3
import sys

def conversion(line, shift=20):
    # assume all have 4 decimal places, e.g.,
    # 123.0000
    #  12.3000
    #   1.2300
    #   0.0002
    f = int(line.replace('.', ''))
    return round(f * 2**(shift)/(10**4))

if __name__ == '__main__':
    for line in sys.stdin:
        print(conversion(line))
