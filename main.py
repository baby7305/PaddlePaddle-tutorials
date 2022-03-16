#%%

def calculate_fee(distance_travelled):
    return 10 + 2 * distance_travelled

for x in [1.0, 3.0, 5.0, 9.0, 10.0, 20.0]:
    print(calculate_fee(x))
    
#%%

import paddle
print("paddle " + paddle.__version__)

#%%

x_data = paddle.to_tensor([[1.], [3.0], [5.0], [9.0], [10.0], [20.0]])
y_data = paddle.to_tensor([[12.], [16.0], [20.0], [28.0], [30.0], [50.0]])

#%%

linear = paddle.nn.Linear(in_features=1, out_features=1)

#%%

w_before_opt = linear.weight.numpy().item()
b_before_opt = linear.bias.numpy().item()

print("w before optimize: {}".format(w_before_opt))
print("b before optimize: {}".format(b_before_opt))

