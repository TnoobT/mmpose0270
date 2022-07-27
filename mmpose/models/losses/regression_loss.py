# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import dsntnn
from ..builder import LOSSES
from ..utils.realnvp import RealNVP


@LOSSES.register_module()
class RLELoss(nn.Module):
    """RLE Loss.

    `Human Pose Regression With Residual Log-Likelihood Estimation
    arXiv: <https://arxiv.org/abs/2107.11291>`_.

    Code is modified from `the official implementation
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow
            learn the residual error distribution.
        q_dis (string): Option for the identity Q(error) distribution,
            Options: "laplace" or "gaussian"
    """

    def __init__(self,
                 use_target_weight=False,
                 size_average=True,
                 residual=True,
                 q_dis='laplace'):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_dis = q_dis

        self.flow_model = RealNVP()

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D*2]): Output regression,
                    including coords and sigmas.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        pred = output[:, :, :2]
        sigma = output[:, :, 2:4].sigmoid()

        error = (pred - target) / (sigma + 1e-9)
        # (B, K, 2)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 2))
        log_phi = log_phi.reshape(target.shape[0], target.shape[1], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], target.shape[1],
                                             2)
        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_dis in ['laplace', 'gaussian', 'strict']
            if self.q_dis == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss

        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """SmoothL1Loss loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.smooth_l1_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight,
                                  target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@LOSSES.register_module()
class WingLoss(nn.Module):
    """Wing Loss. paper ref: 'Wing Loss for Robust Facial Landmark Localisation
    with Convolutional Neural Networks' Feng et al. CVPR'2018.

    Args:
        omega (float): Also referred to as width.
        epsilon (float): Also referred to as curvature.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 omega=10.0,
                 epsilon=2.0,
                 use_target_weight=False,
                 loss_weight=1.):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

        # constant that smoothly links the piecewise-defined linear
        # and nonlinear parts
        self.C = self.omega * (1.0 - math.log(1.0 + self.omega / self.epsilon))

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            pred (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
        """
        delta = (target - pred).abs()
        losses = torch.where(
            delta < self.omega,
            self.omega * torch.log(1.0 + delta / self.epsilon), delta - self.C)
        return torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N,K,D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight,
                                  target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@LOSSES.register_module()
class SoftWingLoss(nn.Module):
    """Soft Wing Loss 'Structure-Coherent Deep Feature Learning for Robust Face
    Alignment' Lin et al. TIP'2021.

    loss =
        1. |x|                           , if |x| < omega1
        2. omega2*ln(1+|x|/epsilon) + B, if |x| >= omega1

    Args:
        omega1 (float): The first threshold.
        omega2 (float): The second threshold.
        epsilon (float): Also referred to as curvature.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 omega1=2.0,
                 omega2=20.0,
                 epsilon=0.5,
                 use_target_weight=False,
                 loss_weight=1.):
        super().__init__()
        self.omega1 = omega1
        self.omega2 = omega2
        self.epsilon = epsilon
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

        # constant that smoothly links the piecewise-defined linear
        # and nonlinear parts
        self.B = self.omega1 - self.omega2 * math.log(1.0 + self.omega1 /
                                                      self.epsilon)

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            batch_size: N
            num_keypoints: K
            dimension of keypoints: D (D=2 or D=3)

        Args:
            pred (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
        """
        delta = (target - pred).abs()
        losses = torch.where(
            delta < self.omega1, delta,
            self.omega2 * torch.log(1.0 + delta / self.epsilon) + self.B)
        return torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K
            dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight,
                                  target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@LOSSES.register_module()
class MPJPELoss(nn.Module):
    """MPJPE (Mean Per Joint Position Error) loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N,K,D]):
                Weights across different joint types.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = torch.mean(
                torch.norm((output - target) * target_weight, dim=-1))
        else:
            loss = torch.mean(torch.norm(output - target, dim=-1))

        return loss * self.loss_weight


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1Loss loss ."""

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.l1_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output regression.
            target (torch.Tensor[N, K, 2]): Target regression.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight,
                                  target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSE loss for coordinate regression."""

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.mse_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output regression.
            target (torch.Tensor[N, K, 2]): Target regression.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight,
                                  target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@LOSSES.register_module()
class BoneLoss(nn.Module):
    """Bone length loss.

    Args:
        joint_parents (list): Indices of each joint's parent joint.
        use_target_weight (bool): Option to use weighted bone loss.
            Different bone types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, joint_parents, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.joint_parents = joint_parents
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

        self.non_root_indices = []
        for i in range(len(self.joint_parents)):
            if i != self.joint_parents[i]:
                self.non_root_indices.append(i)

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K-1]):
                Weights across different bone types.
        """
        output_bone = torch.norm(
            output - output[:, self.joint_parents, :],
            dim=-1)[:, self.non_root_indices]
        target_bone = torch.norm(
            target - target[:, self.joint_parents, :],
            dim=-1)[:, self.non_root_indices]
        if self.use_target_weight:
            assert target_weight is not None
            loss = torch.mean(
                torch.abs((output_bone * target_weight).mean(dim=0) -
                          (target_bone * target_weight).mean(dim=0)))
        else:
            loss = torch.mean(
                torch.abs(output_bone.mean(dim=0) - target_bone.mean(dim=0)))

        return loss * self.loss_weight


@LOSSES.register_module()
class SemiSupervisionLoss(nn.Module):
    """Semi-supervision loss for unlabeled data. It is composed of projection
    loss and bone loss.

    Paper ref: `3D human pose estimation in video with temporal convolutions
    and semi-supervised training` Dario Pavllo et al. CVPR'2019.

    Args:
        joint_parents (list): Indices of each joint's parent joint.
        projection_loss_weight (float): Weight for projection loss.
        bone_loss_weight (float): Weight for bone loss.
        warmup_iterations (int): Number of warmup iterations. In the first
            `warmup_iterations` iterations, the model is trained only on
            labeled data, and semi-supervision loss will be 0.
            This is a workaround since currently we cannot access
            epoch number in loss functions. Note that the iteration number in
            an epoch can be changed due to different GPU numbers in multi-GPU
            settings. So please set this parameter carefully.
            warmup_iterations = dataset_size // samples_per_gpu // gpu_num
            * warmup_epochs
    """

    def __init__(self,
                 joint_parents,
                 projection_loss_weight=1.,
                 bone_loss_weight=1.,
                 warmup_iterations=0):
        super().__init__()
        self.criterion_projection = MPJPELoss(
            loss_weight=projection_loss_weight)
        self.criterion_bone = BoneLoss(
            joint_parents, loss_weight=bone_loss_weight)
        self.warmup_iterations = warmup_iterations
        self.num_iterations = 0

    @staticmethod
    def project_joints(x, intrinsics):
        """Project 3D joint coordinates to 2D image plane using camera
        intrinsic parameters.

        Args:
            x (torch.Tensor[N, K, 3]): 3D joint coordinates.
            intrinsics (torch.Tensor[N, 4] | torch.Tensor[N, 9]): Camera
                intrinsics: f (2), c (2), k (3), p (2).
        """
        while intrinsics.dim() < x.dim():
            intrinsics.unsqueeze_(1)
        f = intrinsics[..., :2]
        c = intrinsics[..., 2:4]
        _x = torch.clamp(x[:, :, :2] / x[:, :, 2:], -1, 1)
        if intrinsics.shape[-1] == 9:
            k = intrinsics[..., 4:7]
            p = intrinsics[..., 7:9]

            r2 = torch.sum(_x[:, :, :2]**2, dim=-1, keepdim=True)
            radial = 1 + torch.sum(
                k * torch.cat((r2, r2**2, r2**3), dim=-1),
                dim=-1,
                keepdim=True)
            tan = torch.sum(p * _x, dim=-1, keepdim=True)
            _x = _x * (radial + tan) + p * r2
        _x = f * _x + c
        return _x

    def forward(self, output, target):
        losses = dict()

        self.num_iterations += 1
        if self.num_iterations <= self.warmup_iterations:
            return losses

        labeled_pose = output['labeled_pose']
        unlabeled_pose = output['unlabeled_pose']
        unlabeled_traj = output['unlabeled_traj']
        unlabeled_target_2d = target['unlabeled_target_2d']
        intrinsics = target['intrinsics']

        # projection loss
        unlabeled_output = unlabeled_pose + unlabeled_traj
        unlabeled_output_2d = self.project_joints(unlabeled_output, intrinsics)
        loss_proj = self.criterion_projection(unlabeled_output_2d,
                                              unlabeled_target_2d, None)
        losses['proj_loss'] = loss_proj

        # bone loss
        loss_bone = self.criterion_bone(unlabeled_pose, labeled_pose, None)
        losses['bone_loss'] = loss_bone

        return losses




# tfj
@LOSSES.register_module()
class DSNTLoss(nn.Module):
    """SmoothL1Loss loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, sigma = 1, mse_weight=1,js_weight = 1, is_dsnt = True):
        super().__init__()
        
        self.mse = dsntnn.euclidean_losses
        self.js = dsntnn.js_reg_losses
        self.avg = dsntnn.average_loss

        self.use_target_weight = use_target_weight
        self.mse_weight = mse_weight
        self.js_weight = js_weight
        self.sigma = sigma

    def forward(self, output, target, heatmap,target_weight=None): # 预测坐标，gt坐标，预测热图， (n,num_joints,h,w)
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            mse_loss = self.mse(output * target_weight,
                                  target * target_weight)
        else:
            mse_loss = self.mse(output, target)
        js_loss = self.js(heatmap, target, sigma_t=self.sigma)

        loss = self.avg(self.mse_weight*mse_loss + self.js_weight*js_loss)

        return loss


@LOSSES.register_module()
class DSNTRLELoss(nn.Module):
    """SmoothL1Loss loss.

    Args:
        rle损失+dsnt损失
    """

    def __init__(self, dsnt_param, rle_param, dsnt_weight, rle_weight):
        super().__init__()
        
        dsnt_use_target_weight = getattr(dsnt_param,'use_target_weight',True)
        sigma = getattr(dsnt_param,'sigma',0.25)
        mse_weight = getattr(dsnt_param,'mse_weight',1)
        js_weight = getattr(dsnt_param,'js_weight',1)
        self.dsnt = DSNTLoss(dsnt_use_target_weight,sigma,mse_weight,js_weight)

        rle_use_target_weight = getattr(rle_param,'use_target_weight',True)
        size_average = getattr(rle_param,'size_average',True)
        residual = getattr(rle_param,'residual',True)
        self.rle = RLELoss(rle_use_target_weight,size_average,residual)
        self.dw = dsnt_weight
        self.re = rle_weight


    def forward(self, output, target, heatmap,target_weight=None):
      
        dsnt_loss = self.dsnt(output[...,:2], target, heatmap, target_weight)
        rle_loss = self.rle(output, target, target_weight)
        loss = self.dw*dsnt_loss + rle_loss*self.re

        return loss


class KLDiscretLoss(nn.Module):
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1) #[B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none')
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1) 
        return loss

    def forward(self, output_x, output_y, target_x, target_y, target_weight):
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:,idx].squeeze()
            coord_y_pred = output_y[:,idx].squeeze()
            coord_x_gt = target_x[:,idx].squeeze()
            coord_y_gt = target_y[:,idx].squeeze()
            weight = target_weight[:,idx,0].squeeze()
            loss += (self.criterion(coord_x_pred,coord_x_gt).mul(weight).mean()) 
            loss += (self.criterion(coord_y_pred,coord_y_gt).mul(weight).mean())
        return loss / num_joints 


@LOSSES.register_module()
class DSNTRLE_SimC_Loss(nn.Module):
    """SmoothL1Loss loss.

    Args:
        rle损失+dsnt损失+simcc损失
    """

    def __init__(self, dsnt_param, rle_param, dsnt_weight, rle_weight, simc_weight):
        super().__init__()
        
        dsnt_use_target_weight = getattr(dsnt_param,'use_target_weight',True)
        sigma = getattr(dsnt_param,'sigma',0.25)
        mse_weight = getattr(dsnt_param,'mse_weight',1)
        js_weight = getattr(dsnt_param,'js_weight',1)
        self.dsnt = DSNTLoss(dsnt_use_target_weight,sigma,mse_weight,js_weight)

        rle_use_target_weight = getattr(rle_param,'use_target_weight',True)
        size_average = getattr(rle_param,'size_average',True)
        residual = getattr(rle_param,'residual',True)
        self.rle = RLELoss(rle_use_target_weight,size_average,residual)
        self.simc = KLDiscretLoss()
    
        self.dw = dsnt_weight
        self.re = rle_weight
        self.sw = simc_weight


    def forward(self, output, target,pred_x,pred_y,target_x,target_y, heatmap,target_weight):
      
        dsnt_loss = self.dsnt(output[...,:2], target, heatmap, target_weight)
        rle_loss = self.rle(output, target, target_weight)
        cimc_loss = self.simc(pred_x, pred_y, target_x, target_y, target_weight)

        loss = self.dw*dsnt_loss + rle_loss*self.re + cimc_loss*self.sw

        return loss