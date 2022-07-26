# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
import dsntnn
import torch
from mmcv.cnn import normal_init,build_upsample_layer

from mmpose.core.evaluation import (keypoint_pck_accuracy,
                                    keypoints_from_regression)
from mmpose.core.post_processing import fliplr_regression
from mmpose.models.builder import HEADS, build_loss


@HEADS.register_module()
class IntegralPoseRegressionHead(nn.Module):
    """Deeppose regression head with fully connected layers.

    "DeepPose: Human Pose Estimation via Deep Neural Networks".

    Args:
        in_channels (int): Number of input channels
        num_joints (int): Number of joints
        loss_keypoint (dict): Config for keypoint loss. Default: None.
        out_sigma (bool): Predict the sigma (the viriance of the joint
            location) together with the joint location. Default: False
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 loss_keypoint=None,
                 out_sigma=False,
                 out_highres = False,
                 with_simcc = False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints

        self.loss = build_loss(loss_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.out_sigma = out_sigma
        self.out_highres = out_highres
        self.with_simcc = with_simcc

        if with_simcc:
            self.mlp_head_x = nn.Linear(32*32, int(256*2))
            self.mlp_head_y = nn.Linear(32*32, int(256*2))

        if out_sigma:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.in_channels, self.num_joints * 2)
            self.conv = nn.Conv2d(self.in_channels, self.num_joints, kernel_size=1, bias=False)
            if self.out_highres:
                self.deconv = self._make_deconv_layer( 2, (256, 16), (4, 4) )
                self.in_channels = in_channels
                self.conv = nn.Conv2d(self.num_joints, self.num_joints, kernel_size=1, bias=False)
                self.fc = nn.Linear(self.in_channels, self.num_joints * 2)
                
        else:
            self.conv = nn.Conv2d(self.in_channels, self.num_joints, kernel_size=1, bias=False)
            if self.out_highres:
                self.deconv = self._make_deconv_layer( 2, (256, 16), (4, 4) )
                self.in_channels = in_channels
                self.conv = nn.Conv2d(self.num_joints, self.num_joints, kernel_size=1, bias=False)
                
    def forward(self, x):
        """Forward function."""
        if isinstance(x, (list, tuple)):
            assert len(x) == 1, ('DeepPoseRegressionHead only supports '
                                 'single-level feature.')
            x = x[0]
        x_copy = x[:]
        if self.out_sigma: # rle
            if self.out_highres:
                x = self.deconv(x)
            unnormalized_heatmaps = self.conv(x) # (64,16,8,8)
            heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps) #  (64,16,8,8)
            # 4. Calculate the coordinates
            coords = dsntnn.dsnt(heatmaps) # (64,16,2)
            global_feature = self.avg(x_copy).reshape(-1,self.in_channels)
            sigma = self.fc(global_feature).reshape(-1,self.num_joints,2)
            coords = torch.cat([coords,sigma],dim = -1)
        else:
            if self.out_highres:
                x = self.deconv(x)
            unnormalized_heatmaps = self.conv(x)
            heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
            # 4. Calculate the coordinates
            coords = dsntnn.dsnt(heatmaps)

        if self.with_simcc:
            b,c,h,w = x.shape
            vec_x = x.view(b,self.num_joints,-1) # (64,16,32*32)
            pred_x = self.mlp_head_x(vec_x)
            pred_y = self.mlp_head_y(vec_x)
            return coords,heatmaps,pred_x,pred_y

        return coords,heatmaps

    def get_loss(self, output, target, heatmap,target_weight,pred_x = None,pred_y= None,target_x= None,target_y= None):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2 or 4]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3
        if self.with_simcc:
            losses['reg_loss'] = self.loss(output, target,pred_x,pred_y,target_x,target_y, heatmap,target_weight)
        else:
            losses['reg_loss'] = self.loss(output, target, heatmap, target_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2 or 4]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        accuracy = dict()

        N = output.shape[0]
        output = output[..., :2]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['acc_pose'] = avg_acc

        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_regression (np.ndarray): Output regression.

        Args:
            x (torch.Tensor[N, K, 2]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        if self.with_simcc:
            output,heatmap,pred_x,pred_y = self.forward(x)
        else:
            output,heatmap = self.forward(x)

        if flip_pairs is not None:
            output_regression = fliplr_regression(
                output.detach().cpu().numpy(), flip_pairs)
        else:
            output_regression = output.detach().cpu().numpy()
        return output_regression

    def decode(self, img_metas, output, **kwargs):
        """Decode the keypoints from output regression.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, >=2]): predicted regression vector.
            kwargs: dict contains 'img_size'.
                img_size (tuple(img_width, img_height)): input image size.
        """
        batch_size = len(img_metas)
        output = output[..., :2]  # get prediction joint locations

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_regression(output, c, s,
                                                   kwargs['img_size'])

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    def init_weights(self):
        normal_init(self.conv, mean=0, std=0.01, bias=0)


    def _get_deconv_cfg(self,deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding


    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)