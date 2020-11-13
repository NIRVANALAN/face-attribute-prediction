python main.py -d /mnt/lustre/yslan/Dataset/CelebA/CelebA \
  -dl /mnt/lustre/yslan/Dataset/CelebA/lfw \
  -c checkpoints/bce_baseline_cos_BS_focal_CropResize_normalize  \
  -pt --sampler balance --focal \
