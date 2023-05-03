# Analogy-Forming Transformers for Few-Shot 3D Parsing
Code for our [ICLR 2023 paper](https://arxiv.org/abs/2304.14382).

## Requirements

We showcase the installation for CUDA 11.1 and torch==1.10.2, which is what we used for our experiments.
We were able to reproduce the results with PyTorch 1.12 and CUDA 11.3 as well.
If you need to use a different version, you can try to modify the following instructions accordingly.

- Create environment: `conda create -n analogical python=3.8 anaconda`

- Activate environment: `conda activate analogical`

- Install torch: `pip install -U -f torch==1.10.2 torchvision==0.11.3 --extra-index-url https://download.pytorch.org/whl/cu111`

- Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network: `sh init.sh`

- Other requirements: `pip install -r requirements.txt`

  

For the visualizations we show in the paper, install [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). This step can be skipped if you don't want to run the visualization code. In this case, you need to comment lines 4-14 in src/tools.py.
For CUDA 11.1, follow these steps:

- Install PyTorch3D dependencies: `conda install -c fvcore -c iopath -c conda-forge fvcore iopath`

- Install CUB: `conda install -c bottler nvidiacub`

- Install PyTorch3D: `pip install "git+https://github.com/facebookresearch/pytorch3d.git"`

  

## Data

- [Download PartNet](https://www.shapenet.org/download/parts). We only need ins_seg_h5.zip. Unzip. Then let PATH_TO_ins_seg_h5 be the path to the unzipped folder `ins_seg_h5/`, found inside ins_seg_h5.zip.

- Clone `https://github.com/daerduoCarey/partnet_dataset` to PATH_TO_PARTNET_REPO

- Run `python prepare_partnet.py --merging_data_path PATH_TO_PARTNET_REPO/stats/after_merging_label_ids/ --in_path PATH_TO_ins_seg_h5 --out_path PATH_TO_PARTNET`, where PATH_TO_PARTNET is the path to store the processed annotation files (you can define this path).

  

## Run
Detailed scripts and checkpoints coming soon (ETA May 5th, 2023).

