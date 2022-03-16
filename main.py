#%%

import paddle
import paddle.nn.functional as F
import numpy as np

print(paddle.__version__)

#%%

a = paddle.randn([4, 2])
b = paddle.arange(1, 3, dtype='float32')

print(a)
print(b)

c = a + b
print(c)

d = paddle.matmul(a, b)
print(d)

