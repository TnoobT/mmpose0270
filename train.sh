
CUDA_VISIBLE_DEVICES=4,5,6,7,  bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/deeppose/mpii/29_reg_shufflenetv2_mpii_256x256_dsnt_rle_highres_simc.py 4
sleep 5
CUDA_VISIBLE_DEVICES=4,5,6,7,  bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/deeppose/mpii/30_reg_shufflenetv2_mpii_256x256_dsnt_rle_highres_simc.py 4
sleep 5
CUDA_VISIBLE_DEVICES=4,5,6,7,  bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/deeppose/mpii/31_reg_shufflenetv2_mpii_256x256_dsnt_rle_highres_simc.py 4
sleep 5
CUDA_VISIBLE_DEVICES=4,5,6,7,  bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/deeppose/mpii/32_reg_shufflenetv2_mpii_256x256_dsnt_rle_highres_simc.py 4