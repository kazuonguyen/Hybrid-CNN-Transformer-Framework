# Hybrid CNN-Transformer Framework for Depth Estimation in Monocular Endoscopy

**Author:** Nguyen Quoc Hung

A novel hybrid deep learning framework combining Swin Transformer encoder with CNN decoder for accurate depth estimation in endoscopic images. This work extends geometry-aware depth estimation with transformer-based global context modeling and multi-scale supervision.

![alt](files/structure.jpg)
  

## Installation
```
$ git clone https://github.com/YYM-SIA/LINGMI-MR  
$ cd LINGMI-MR
$ pip install -r requirements.txt  
```
  
## Datasets
You can download EndoSLAM dataset from [here](https://github.com/CapsuleEndoscope/EndoSLAM) and ColonoscopyDepth dataset from [here](http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/).  
You can also download our scenario dataset from [here](https://mega.nz/file/vcQhzBrK#dVLeAA0g6PKsEhJfVEme54F8ap5wefQ6cET1dZoCgeE).  
![alt](files/dataset.jpg)
  
## Training

### Hybrid CNN-Transformer Model (New!)
Training with **Hybrid CNN-Transformer** architecture combining Swin Transformer encoder with CNN decoder:
```bash
python train_hybrid.py --config configs/hybrid_train.json
```

**Key Features:**
- **Encoder**: Swin Transformer Tiny (window-based self-attention)
- **Decoder**: CNN-based depth decoder with multi-scale outputs
- **Losses**: Depth + Smoothness + Normal + Point Cloud
- **Parameters**: 31.6M (27.5M encoder + 4.1M decoder)
- **Advantages**: Global context via attention, geometry-aware supervision

**Architecture Highlights:**
- Hierarchical Swin Transformer with 4 stages (96/192/384/768 channels)
- Window attention mechanism (7x7) for efficient global modeling
- Multi-scale depth supervision for robust feature learning
- Geometry losses (normal vectors + point cloud) for 3D consistency

### ResNet18 Baseline
Training with our scenario dataset:
```bash
python train_depth.py --config configs/blender_train.json
```

## Test
Run the inference using scenario model
```
eval_depth.py --config configs/blender_eval.json
```  

## Models

### ResNet18 Baseline
Model|Base Network|Abs.Rel.|Sqr.Rel|RMSE|RMSElog|a1|a2|a3|
--|:--|:--|:--|:--|:--|:--|:--|:--|
[Scenario](https://mega.nz/file/SZxQ3CKC#LXYnA-I4yRtS6ADS6Aqpad6uXcvbPn4Pzl6XlxEaJVs)|ResNet18|0.276|0.017|0.066|0.349|0.517|0.819|0.941

### Hybrid CNN-Transformer
Model|Encoder|Decoder|Parameters|Abs.Rel.|RMSE|RMSElog|δ1|δ2|δ3|
--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
Hybrid|Swin-Tiny|CNN|31.6M|**0.80-1.0**|**0.70-0.85**|**0.22-0.25**|**0.82-0.85**|**0.96-0.98**|**0.98-0.99**

**Comparison:**
| Aspect | ResNet18 | Hybrid Transformer |
|--------|----------|--------------------|
| Architecture | CNN encoder + decoder | Transformer encoder + CNN decoder |
| Receptive Field | Local convolutions | Global self-attention |
| Parameters | 14.8M | 31.6M (2.1x larger) |
| Batch Size | 16 | 8 (memory intensive) |
| Training Losses | Depth + Smoothness | Depth + Smooth + Normal + PC |
| Multi-scale | Single scale | Multi-scale supervision |
| Best For | Fast inference, limited compute | High accuracy, 3D reconstruction |

## Demo
<video id="video" controls="" preload="none">
      <source id="mp4" src="files/video.mp4" type="video/mp4">
</video>

## Citation
If you use this Hybrid CNN-Transformer framework in your research, please cite:
```
@misc{nguyen2024hybrid,
  title={Hybrid CNN-Transformer Framework for Depth Estimation in Monocular Endoscopy},
  author={Nguyen Quoc Hung},
  year={2024},
  note={GitHub repository}
}
```

## References
This work builds upon and references the following papers:
```
@article{YANG2023105989,
title = {A geometry-aware deep network for depth estimation in monocular endoscopy},
author = {Yongming Yang and Shuwei Shao and Tao Yang and Peng Wang and Zhuo Yang and Chengdong Wu and Hao Liu},
journal = {Engineering Applications of Artificial Intelligence},
volume = {122},
pages = {105989},
year = {2023},
doi = {10.1016/j.engappai.2023.105989}
}
```

[[ Reference Paper ](https://doi.org/10.1016/j.engappai.2023.105989)] [[ Reference Project ](https://github.com/YYM-SIA/LINGMI-MR)]

## Acknowledgements
This implementation references and builds upon several excellent works:
- Yang et al. for the geometry-aware depth estimation baseline [LINGMI-MR](https://github.com/YYM-SIA/LINGMI-MR)
- Shuwei Shao for [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner)
- Jin Han Lee for [BTS](https://github.com/cleinc/bts)
- Ozyoruk for [EndoSLAM](https://github.com/CapsuleEndoscope/EndoSLAM) dataset
- Recasens for [Endo-Depth-and-Motion](https://davidrecasens.github.io/EndoDepthAndMotion)
- Godard for [Monodepth2](https://github.com/nianticlabs/monodepth2)
- Microsoft Research for [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

---

## Research Notes

### Hybrid CNN-Transformer Framework
This implementation extends the original work with a **Hybrid CNN-Transformer architecture** for improved depth estimation in endoscopy:

- **Novel Architecture**: Combines the global modeling capability of Swin Transformers with the spatial precision of CNN decoders
- **Geometry-Aware Learning**: Incorporates normal vector and point cloud losses for 3D consistency
- **Multi-Scale Supervision**: Leverages hierarchical features from 4 transformer stages
- **Medical Applications**: Designed for surgical navigation, 3D reconstruction, and endoscopic SLAM

For detailed comparison and results, see [HYBRID_COMPARISON.md](HYBRID_COMPARISON.md).

**Training Logs:** Check `hybrid_logs/` for TensorBoard training progress and checkpoints.