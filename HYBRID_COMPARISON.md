# Hybrid CNN-Transformer vs ResNet18 Comparison

## 1. ResNet18 Baseline

### Architecture
- Encoder: ResNet18 + Spatial Attention Block
- Decoder: CNN-based depth decoder
- Parameters: 14.8M total (11.7M encoder, 3.1M decoder)
- Output: Single-scale depth map (320x320)

### Training Configuration
- Learning rate: 0.01 (exponential decay 0.88)
- Optimizer: Adam
- Batch size: 16
- Losses: L1 depth + edge-aware smoothness
- Loss weights: depth=1.0, smoothness=0.5

## 2. Hybrid CNN-Transformer

### Architecture
- Encoder: Swin Transformer Tiny (window size 7x7)
  - 4 stages with 2/2/6/2 blocks
  - Multi-head attention: 3/6/12/24 heads
  - Hierarchical features: 96/192/384/768 channels
- Decoder: CNN-based depth decoder
- Parameters: 31.6M total (27.5M encoder, 4.1M decoder)
- Output: Multi-scale depth maps

### Training Configuration
- Learning rate: 0.0001 (exponential decay 0.9)
- Optimizer: Adam
- Batch size: 8
- Losses: L1 depth + smoothness + normal + point cloud
- Loss weights: depth=1.0, smooth=0.5, pc=0.25, normal=0.25

## 3. Key Differences

| Aspect | ResNet18 | Hybrid Transformer |
|--------|----------|--------------------|
| Encoder | CNN-based | Transformer-based |
| Receptive field | Local | Global (attention) |
| Parameters | 14.8M | 31.6M |
| Batch size | 16 | 8 |
| Learning rate | 0.01 | 0.0001 |
| Supervision | Single-scale | Multi-scale |
| Geometry loss | No | Yes (normal + PC) |
| Training speed | 17.2 it/s | Slower |
| Memory usage | 8 GB | Higher |

## 4. Advantages and Trade-offs

### Hybrid Advantages
- Global context via self-attention
- Multi-scale supervision for robust features
- Geometry-aware losses (normal + point cloud)
- Hierarchical feature learning
- Better for 3D reconstruction tasks

### Hybrid Disadvantages
- 2x more parameters (31.6M vs 14.8M)
- Higher computational cost
- Requires more GPU memory
- Slower training and inference
- Lower batch size (8 vs 16)

## 5. Expected Performance

| Metric | ResNet18 | Hybrid (Expected) |
|--------|----------|-------------------|
| Abs.Rel | 1.163 | 0.8-1.0 |
| RMSE | 0.918 | 0.7-0.85 |
| RMSElog | 0.265 | 0.22-0.25 |
| δ1 | 0.799 | 0.82-0.85 |
| δ2 | 0.952 | 0.96-0.98 |
| δ3 | 0.975 | 0.98-0.99 |

## 6. Research Alignment

### Topic: Hybrid CNN-Transformer for Depth Estimation in Endoscopy

**Implementation Coverage:**
- Hybrid architecture: Transformer encoder + CNN decoder
- Deep learning-based depth estimation
- Geometry-aware training (normal + point cloud losses)
- 3D reconstruction capability (depth to point cloud)
- Endoscopy-specific optimization (Unity Colon dataset)

**Applications:**
- Surgical planning and navigation
- Minimally invasive surgery assistance
- Medical education and training
- Foundation for endoscopic SLAM systems
