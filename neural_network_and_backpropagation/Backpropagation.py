#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
def sigm(x):
    return 1 / (1 + np.exp(-x))
def sigm_d(sig):
    return sig * (1 - sig)
input = 2
hidden = 3
output = 3
inputs = np.random.uniform(0.01, 1.0,input)
targets = np.random.uniform(0.1, 1.0,output)
eta = np.random.uniform(0.01, 1.0)
w = np.random.uniform(0.1, 1.0,(hidden,input))
theta = np.random.uniform(0.1, 1.0,(output,hidden))

i=0
l=float('inf')
while l>1e-16:
    h = np.dot(w,inputs)
    Oh = sigm(h)
    o = np.dot(theta,Oh)
    Oj = sigm(o)

    l = 0.5*np.sum((targets - Oj)**2)
    err_out = (Oj - targets)*sigm_d(Oj)
    del_theta = np.outer(err_out,Oh)
    err_hidden = np.dot(theta.T, err_out)*sigm_d(Oh)
    del_w = np.outer(err_hidden,inputs)

    theta = theta - eta*del_theta
    w = w - eta*del_w

    print(f"i {i}")
    print("Inputs:",inputs)
    print("Targets:",targets)
    print("Hidden Outputs (Oh):",Oh)
    print("Outputs (Oj):",Oj)
    print("Loss:",l)
    print("New w:",w)
    print("New theta:",theta)
    print("\n")
    i+=1
print("Total Iterations:",i)
print("Total Loss:",l)


# In[ ]:




