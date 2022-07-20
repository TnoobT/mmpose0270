import scipy
import cv2
import torch
import numpy as np
torch.manual_seed(12345)
from torch import nn
import dsntnn
import torch.optim as optim
 
class FCN(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3,stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3,stride=4, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3,stride=4, padding=1),
        )
 
    def forward(self, x):
        return self.layers(x)
class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations):
        super().__init__()
        self.fcn = FCN()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1, bias=False)
 
    def forward(self, images):
        # 1. Run the images through our FCN
        fcn_out = self.fcn(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)
 
        return coords, heatmaps
 

img = cv2.imread('./test.jpg')
img = cv2.resize(img,[256,256])
image_size = [256, 256]

anno = [[170,187],[206,228],[255,192],[287,130],[161,236],[112,222],[67,207],[204,374],[206,460],[238,567],[152,376],[148,459],[158,560]]

# for kk in range(len(anno)):                   
#     cv2.circle(img,(int(anno[kk][0]),int(anno[kk][1])),2,(255,255,0))
# cv2.imshow("s",img)
# cv2.waitKey(0)

tmp_img = torch.from_numpy(img).permute(2, 0, 1).float()   # torch.Size([3, 640, 360])
input_tensor = tmp_img.div(255).unsqueeze(0)    # torch.Size([1, 3, 640, 360])
input_var = input_tensor
# input_var = torch.repeat_interleave(input_var,2,0)

target_tensor = torch.Tensor([anno])    # # shape = [1, 13, 2]
target_tensor = (target_tensor * 2 - 1) / torch.Tensor(image_size) - 1  # shape = [1, 13, 2], 归一化到[-1,1]
target_var = target_tensor
# target_var = torch.repeat_interleave(target_var,2,0)

model = CoordRegressionNetwork(n_locations=13)   # n_locations=keypoint num=1
optimizer = optim.RMSprop(model.parameters(), lr=1.0e-4)
 
for i in range(4000):
    # Forward pass
    coords, heatmaps = model(input_var)
    # coords:shape=[1, 1, 2], value=[[[0.0323, 0.0566]]]; heatmaps:shape=[1, 1, 40, 40]
    # Per-location euclidean losses
    euc_losses = dsntnn.euclidean_losses(coords, target_var)
    # Per-location regularization losses
    reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=2.0)
    # Combine losses into an overall loss
    loss = dsntnn.average_loss(euc_losses + reg_losses)
 
    # Calculate gradients
    optimizer.zero_grad()
    loss.backward()
 
    # Update model parameters with RMSprop
    optimizer.step()
    print('loss: {}'.format(loss.item()))

# Predictions after training

pred = ((coords[0].detach().cpu().numpy() + 1)*image_size + 1)//2

for kk in range(len(pred)):                   
    cv2.circle(img,(int(pred[kk][0]),int(pred[kk][1])),3,(0,255,255))
cv2.imshow("pred",img)
cv2.waitKey(0)