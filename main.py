import torch
import torch.nn
import time
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
in_row, in_f, out_f = 85600, 20240, 20480
loop_times = 10

s = time.time()
tensor = torch.randn(in_row, in_f).cuda()
l_trans = torch.nn.Linear(in_f, out_f).cuda()
for _ in range(loop_times):
    l_trans(tensor)
torch.cuda.synchronize()
print('CUDA take time:', time.time()-s)
