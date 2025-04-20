<p align="center">

  <h1 align="center">AAAI25' FIRM: Flexible Interactive Reflection reMoval</h1>
  
</p>
<p align="center">

### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks and [HINet](https://github.com/megvii-model/HINet) 

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
git clone https://github.com/megvii-research/NAFNet
cd NAFNet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Train
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/REDS/NAFNet-width64.yml

### Datasets
Please download the datasets with contrastive masks from [OneDrive](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21119074r_connect_polyu_hk/EZKRZU1E1cNBpYCSUSIM7mABLHazetKJDZKWkd9UfyMPCw?e=yt9Ssh)

### Citations
If our work helps your research or work, please consider citing NAFNet.

```
@inproceedings{chen2025firm,
  title={FIRM: Flexible Interactive Reflection ReMoval},
  author={Chen, Xiao and Jiang, Xudong and Tao, Yunkang and Lei, Zhen and Li, Qing and Lei, Chenyang and Zhang, Zhaoxiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={2},
  pages={2230--2238},
  year={2025}
}
```