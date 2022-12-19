import numpy as np

init_np = np.random.rand(30)
init_np_mean = np.mean(init_np)
init_np_var = np.var(init_np)
M = init_np.shape[0]

print('init numpy is: '+str(init_np))
print('init num of numpy is: '+str(M))
print('init mean is: '+str(init_np_mean))
print('init vari is: '+str(init_np_var))

add = np.random.randn(5000)*100
N = add.shape[0]
add_np_mean = np.mean(add)
add_np_var = np.var(add)

delta_mean = (add_np_mean-init_np_mean)*N/(M+N)
mean_new = init_np_mean+delta_mean
delta_var = (N*((add_np_mean-mean_new)**2)-N*(init_np_var-add_np_var)+M*(delta_mean**2))/(M+N)
var_new = init_np_var+delta_var
init_np = np.concatenate((init_np,add),axis=0)

print('#################################')
print('new numpy is: '+str(init_np))
print('new num of numpy is: '+str(M+N))
print('new mean is: '+str(np.mean(init_np))+' our mean is: '+str(mean_new))
print('new var is: '+str(np.var(init_np))+' our var is: '+str(var_new))