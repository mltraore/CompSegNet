# CompSegNet: An enhanced U-shaped architecture for nuclei segmentation in H&amp;E histopathology images

## 1. Abstract
<div align="justify"> 
In histopathology, nuclei within images hold vital diagnostic information. Automated segmentation of nuclei can alleviate pathologists' workload and enhance diagnostic accuracy. Although U-Net-based methods are prevalent, they face challenges like overfitting and limited field-of-view. This paper introduces a new U-shaped architecture (CompSegNet) for nuclei segmentation in H&amp;E histopathology images by developing enhanced convolutional blocks and a Residual Bottleneck Transformer (RBT) block. The proposed convolutional blocks are designed by enhancing the Mobile Convolution (MBConv) block through a receptive fields enlargement strategy, which we referred to as the Zoom-Filter-Rescale (ZFR) strategy and a global context modeling based on the global context (GC) Block;  and the proposed RBT block is developed by incorporating the Transformer encoder blocks in a tailored manner to a variant of the Sandglass block. Additionally, a noise-aware stem block and a weighted joint loss function are designed to improve the overall segmentation performance. The proposed CompSegNet outperforms existing methods quantitatively and qualitatively, achieving a competitive AJI score of 0.705 on the MoNuSeg 2018 dataset, 0.72 on the CoNSeP dataset, and 0.779 on the CPM-17 dataset while maintaining a reasonable parameter count.
</div>

## 2. Architecture
### 2.1 Overall Architecture
![CompSegNet-architecture](misc/compsegnet_arch.png)

### 2.2 Architecture Specifications
![CompSegNet-arch-specs](misc/compsegnet_specs.png)

### 2.3 Building Blocks
#### CompSeg (CSeg) Block
<div align="justify"> 
  
- Enhances MBConv efficiency through the Zoom-Filter-Rescale (ZFR) strategy.
- Involves:
  - Large stride at the entry expansion layer (*Zooming*).
  - Large kernel-sized depthwise convolution operation (*Zooming and Filtering*).
  - Rescaling spatial size using bilinear upsampling to compensate for downsampling (*Rescaling*).
- Increases receptive fields within MBConv by 8×.
- Decreases inference latency by approximately 30%.

</div>

![MBConv-vs-CSeg](misc/mbconv_vs_cseg.png)

#### Extended CompSeg (ECSeg) Block
<div align="justify">

- Extends the CSeg block by incorporating an improved global context (IGC) block for robust global context modeling.
- Increases AJI score of the nuclei segmentation on MoNuSeg 2018 dataset in a lightweight network by approximately 1.9%.
- Achieves improved performance with only a minimal computational cost.

</div>

![IGC-and-ECSeg](misc/igc_and_ecseg.png)

#### Residual Bottleneck Transformer (RBT) Block
<div align="justify">
  
- Incorporates Transformer encoder blocks in a tailored manner to a variant of the Sandglass block.
- Leverages strengths of both convolutional and attention-based mechanisms.
- Effectively captures fine-grained spatial information and comprehensive long-range dependencies.

</div>

![RBT Block](misc/rbt_block.png)

#### Noise-aware stem (Block) Block
<div align="justify">
  
- Stem block 
- Enhances low-level feature extraction while mitigating the impact of noisy features in the model's initial skip connections.  
  
</div>

![NAS Block](misc/nas_block.png)


