import torch
def soft_argmax(value, length):
    size = length
    # 定义一个一维的长度为10的分布
    a = torch.zeros((size, ))

    # 在第8项上设置响应
    a[0] = value
    a[1] = -value
    # 进行softmax归一化
    softmax_res = a.softmax(0)
    print('value: {}, after softmax:{}\n'.format(value,softmax_res[:3]))

    # 求期望值
    lin = torch.tensor([x for x in range(size)])
    expectation = (lin * softmax_res).sum()
    # print('expectation:\n', expectation)
    return expectation

length = 512
for x in  range(512):
    expect = soft_argmax(x, length)
    # err += (idx - expect)**2
    # print(x, err)