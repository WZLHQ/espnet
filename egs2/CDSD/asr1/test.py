import torch
import time

c=256
k=3
x1=torch.rand(64,c,101).to("cuda:0")

conv1=torch.nn.Conv1d(c,c,k,1,(k-1)//2,groups=c).to("cuda:0")

t1=time.time()
for _ in range(24):
    conv1(x1)
t2=time.time()
print(t2-t1)


# c=256
# x1=torch.rand(64,101,c).to("cuda:0")

# conv1=torch.nn.Linear(c,c).to("cuda:0")

# t1=time.time()
# for _ in range(100):
#     conv1(x1)
# t2=time.time()
# print(t2-t1)