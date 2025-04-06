# VecSetX
Following the introduction of [VecSet](https://arxiv.org/abs/2301.11445), extensive work has been done to propose enhancements. This project is designed to incorporate these novel designs and to provide unifed framework for VecSet-based representations.
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Installation
```bash
conda create -y -n vecset python=3.11 -y
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

conda install cuda-nvcc=12.4 -c nvidia -y
conda install libcusparse-dev -y
conda install libcublas-dev -y
conda install libcusolver-dev -y
conda install libcurand-dev -y # torch_cluster

pip install flash-attn --no-build-isolation
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install tensorboard
pip install einops
pip install trimesh
pip install tqdm
pip install PyMCubes
```

## Training Example
16 GPUs (4 GPUs with accum_iter 4)
```bash
torchrun \
    --nproc_per_node=4 \
    main_ae.py \
    --accum_iter=4  \
    --model learnable_vec1024x16_dim1024_depth24_nb \
    --output_dir output/ae/learnable_vec1024x16_dim1024_depth24_sdf \
    --log_dir output/ae/learnable_vec1024x16_dim1024_depth24_sdf \
    --num_workers 24 \
    --point_cloud_size 8192 \
    --batch_size 16 \
    --epochs 500 \
    --warmup_epochs 1 --blr 5e-5 --clip_grad 1
```

## Model Descriptions
The base model design is from [VecSet](https://arxiv.org/abs/2301.11445).
I have incorporated the following features list:
- [x] Faster training with [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [x] Normalized Bottleneck (NBAE) from [LaGeM](https://arxiv.org/abs/2410.01295). No need to tune the KL weight anymore!
- [x] SDF regression instead of occupancy classification suggested by [TripoSG](https://arxiv.org/abs/2502.06608). For now, I only use Eikonal regularization.

I am planning to incorporated the following features:
- [ ] Edge sampling from [Dora-VAE](https://arxiv.org/abs/2412.17808)
- [ ] Multiresolution training from [CLAY](https://arxiv.org/abs/2406.13897)
- [ ] Compact autoencoder from [COD-VAE](https://arxiv.org/abs/2503.08737)
- [ ] Quantized bottleneck (VQ).
- [ ] (Start an issue if you have any ideas!)

## Checkpoints
The following models will be released in this [link](https://huggingface.co/Zbalpha/VecSetX):
- `point_vec1024x32_dim1024_depth24_sdf_nb`: Point Queries, 24 layers, 1024-channel attentions, 1024x32 normalized bottleneck, SDF regression with Eikonal regularizer
<!-- - `point_vec1024x16_dim1024_depth24_sdf_nb`: Point Queries, 24 layers, 1024-channel attentions, 1024x16 normalized bottleneck, SDF regression with Eikonal regularizer -->
- `learnable_vec1024x32_dim1024_depth24_sdf_nb`: Learnable Queries, 24 layers, 1024-channel attentions, 1024x32 normalized bottleneck, SDF regression with Eikonal regularizer
<!-- - `learnable_vec1024x16_dim1024_depth24_sdf_nb`: Learnable Queries, 24 layers, 1024-channel attentions, 1024x16 normalized bottleneck, SDF regression with Eikonal regularizer -->
- `learnable_vec1024_dim1024_depth24_sdf`: Learnable Queries, 24 layers, 1024-channel attentions, 1024x1024 bottleneck, SDF regression with Eikonal regularizer
- (Other models are training!)

## Other minor adjustments
- Removed layernorm on KV suggested by Youkang Kong
- Added layernorm before final output layer.
- Added zero initialization on the final output layer.
- Added random rotations as the data augmentations as in [LaGeM](https://arxiv.org/abs/2410.01295).
- Adjusted code for latest version of PyTorch.

## If you are using this repository in your projects, consider citing the related papers.