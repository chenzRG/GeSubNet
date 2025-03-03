# GeSubNet.

This is an official implementation of paper "GeSubNet: Gene Interaction Inference for Disease Subtype Network Generation". 

## :fire: News

Our work was selected as Oral Presentation at ICLR 2025 Oral!

# Dependencies:
GeSubNet is built based on PyTorch.
You can install PyTorch following the instructions in [PyTorch](https://pytorch.org/get-started/locally/). For example:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
After ensuring that PyTorch is installed correctly, you can install other dependencies via:

```bash
pip install -r requirements.txt
```

# About data:
We have prepared datasets for four cancers: BRCA, GBM, OV, and LGG. The dataset is in the in the directory ```./data/```.

# Training:
For GeSubNet:
The main code is ```./GeSubNet_main.py```.

The available models for Patient-M and Graph-M are in the directory ```./model/```.

The utils needed for training are ```./utils.py```.

Log files will be generated and updated in  ```./log/``` during the training process.

# Case Analysis:
More results of the BRCA case analysis are in the directory ```./Case Analysis/```.

## Citation
If you find our work useful in your research, please consider citing:
```
@misc{gesubnetICLR2025,
      title={GeSubNet: Gene Interaction Inference for Disease Subtype Network Generation}, 
      author={Ziwei Yang and Zheng Chen and Xin Liu and Rikuto Kotoge and Peng Chen and Yasuko Matsubara and Yasushi Sakurai and Jimeng Sun},
      year={2024},
      eprint={2410.13178},
      archivePrefix={arXiv},
      primaryClass={cs.LG} 
}

```
