# SPORT BIKES PRICE PREDICTION PROJECT ( MACHINE LEARNING )

import numpy as np
import matplotlib.pyplot as plt
import math, copy

# 1st Bike: Kawasaki Ninja H2R
# 2nd Bike: KTM RC 200
# 3rd Bike: Apache RR 310
# 4th Bike: Hayabusa
# 5th Bike: BMW G310 GS

# features = [cc, Mileage, Max Torque, Max Power, Fuel Tank Cap.]
x_train = np.array([[998.0, 15.0, 165.00, 322.0, 17.0],[199.5, 35.0, 19.50, 25.00, 13.7],[312.2, 30.0, 27.30, 34.00, 11.0],[1340.0, 11.0, 150.0, 187.00, 20.0],[313.0, 30.0, 28.00, 33.50, 11.0]])
# price in lakhs
y_train = np.array([79.9, 2.15, 2.65, 16.9, 2.9])

# defining the cost function
def compute_cost(x, y, w, b):
    
    m = len(x)
    cost = 0
    for i in range(m):
        f_wb = np.dot(w,x[i]) + b
        cost += (f_wb - y[i])**2
        
    cost = cost/(2*m)
    return cost

# defining the gradient function
def compute_gradient(x, y, w, b):
    
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        f_wb = np.dot(w,x[i]) + b
        for j in range(n):
            dj_dw[j] += (f_wb - y[i])*x[i,j]
        dj_db += (f_wb - y[i])
        
    dj_dw = dj_dw/m
    dj_db = dj_db/n
    
    return dj_dw, dj_db

# defining the Gradient Descent Algorithm
def gradient_descent(x, y, w_init, b_init, alpha, num_iters, cost_func, grad_func):
    
    w = copy.deepcopy(w_init)
    b = b_init
    
    for i in range(num_iters):
        
        dj_dw, dj_db = grad_func(x, y, w, b)
        
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
    # printing the updation of values of w,b
        if i%math.ceil(num_iters/10) == 0:
            print(f"Iteration: {i}, w: {w}, b: {b}")
            
    return w,b

# declaring the parameters 
w_in = np.zeros((5,))
b_in = 0
alpha = 5.0e-8
iters = 100000

w_final, b_final = gradient_descent(x_train, y_train, w_in, b_in, alpha, iters, compute_cost, compute_gradient)
print(f"\nFinal w: {w_final}, b:{b_final}")

def cost_v_iters(x, y, w_init, b_init, alpha, num_iters, cost_func, grad_func):
    
    j_list=[]
    i_list=[]
    w = copy.deepcopy(w_init)
    b = b_init
    
    for i in range(num_iters):
        
        dj_dw, dj_db = grad_func(x, y, w, b)
        
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
    # printing the updation of values of w,b
        if i%math.ceil(num_iters/10) == 0:
            j_list.append(cost_func(x, y, w, b))
            i_list.append(i)
            
    return j_list, i_list

f_j_list, f_i_list = cost_v_iters(x_train, y_train, w_in, b_in, alpha, iters, compute_cost, compute_gradient)

j_array = np.array(f_j_list)
i_array = np.array(f_i_list)

# Plotting the cost function vs iteration graph (decreasing graph) for the verification of model
plt.plot(i_array, j_array)
plt.style.use('seaborn')
plt.title('Cost vs Iterations Graph')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

features = []

cc = float(input('ENTER CC: '))
features.append(cc)
mileage = float(input(('ENTER MILEAGE: ')))
features.append(mileage)
m_t = float(input(('ENTER MAX TORQUE: ')))
features.append(m_t)
m_p = float(input(('ENTER MAX POWER: ')))
features.append(m_p)
f_t_c = float(input(('ENTER FUEL TANK CAP: ')))
features.append(f_t_c)

f = np.array(features)
price = np.dot(f,w_final) + b_final

print(f"Approx price of bike : {price}")
              
              