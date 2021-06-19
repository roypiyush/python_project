#!/usr/bin/env python3.4

import sys

n, k = input().strip().split(' ')
n, k = [int(n), int(k)]
x = sorted([int(x_temp) for x_temp in input().strip().split(' ')])

x[0] = 0
for i in range(0, n):
    x[i] = x[i] - x[i - 1]

count = 0
i = 0
last_placement = -1

while i < n:
    if abs(x[i]) + k < abs(x[i + 1]):
        x[i + 1] = -x[i + 1]
    elif abs(x[i]) + k == abs(x[i + 1]):
        x[i + 1] = -x[i + 1]
    elif abs(x[i]) + k > abs(x[i + 1]):
        x[i] = -x[i]

    if x[i] < 0:
        count = count + 1
        x[i] = abs(x[i])
        last_placement = i

print (count)