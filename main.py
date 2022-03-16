#%%

def calculate_fee(distance_travelled):
    return 10 + 2 * distance_travelled

for x in [1.0, 3.0, 5.0, 9.0, 10.0, 20.0]:
    print(calculate_fee(x))
    
