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

#%%

total_data, batch_size, input_size, hidden_size = 1000, 64, 128, 256

x_data = np.random.randn(total_data, input_size).astype(np.float32)
y_data = np.random.randn(total_data, 1).astype(np.float32)

model = MyModel(input_size, hidden_size)

loss_fn = paddle.nn.MSELoss(reduction='mean')
optimizer = paddle.optimizer.SGD(learning_rate=0.01, 
                                 parameters=model.parameters())

for t in range(200 * (total_data // batch_size)):
    idx = np.random.choice(total_data, batch_size, replace=False)
    x = paddle.to_tensor(x_data[idx,:])
    y = paddle.to_tensor(y_data[idx,:])
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    if t % 200 == 0:
        print(t, loss.numpy())

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

