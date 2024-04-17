import numpy as np

a = [1,2,3,4,5,6,7,8,10]
a= np.array(a)

index = (np.abs(a - 9)).argmin()
print(index)