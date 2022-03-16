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

#%%

a = paddle.to_tensor(np.array([1, 2, 3]))
b = paddle.to_tensor(np.array([4, 5, 6]))

for i in range(10):
    r = paddle.rand([1,])
    if r > 0.5:
        c = paddle.pow(a, i) + b
        print("{} +> {}".format(i, c.numpy()))
    else:
        c = paddle.pow(a, i) - b
        print("{} -> {}".format(i, c.numpy()))

#%%

class MyModel(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size):
        super(MyModel, self).__init__()
        self.linear1 = paddle.nn.Linear(input_size, hidden_size)
        self.linear2 = paddle.nn.Linear(hidden_size, hidden_size)
        self.linear3 = paddle.nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)

        if paddle.rand([1,]) > 0.5: 
            x = self.linear2(x)
            x = F.relu(x)

        x = self.linear3(x)
        
        return x     

