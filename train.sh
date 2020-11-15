python main.py -d /mnt/lustre/yslan/Dataset/CelebA/CelebA \
  -dl /mnt/lustre/yslan/Dataset/CelebA/lfw \
  -pt \
  -c checkpoints/cos_BS_focal_256_WD_1e-4_crop \
  --sampler uniform \
  --focal \
  --evaluate_lfw \
  --resume ./checkpoints/cos_BS_focal_256_WD_1e-4_crop/checkpoint.pth.tar \
#  --lr 0.01 --epochs 40 --ft
#  --weight-decay 2e-4
