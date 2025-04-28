# Low-Light Aerial Imaging With Color and Monochrome Cameras.[TGRS 2025]
## This repository is the official implementation of TGRS 2025 "Low-Light Aerial Imaging With Color and Monochrome Cameras" [paper](https://ieeexplore.ieee.org/abstract/document/10948510). 

## Abstract
Aerial imaging aims to produce well-exposed images
with rich details. However, aerial photography may encounter
low-light conditions during dusk or dawn, as well as on cloudy
or foggy days. In such low-light scenarios, aerial images often
suffer from issues such as underexposure, noise, and color
distortion. Most existing low-light imaging methods struggle
with achieving realistic exposure and retaining rich details. To
address these issues, we propose an Aerial Low-light Imaging
with Color-monochrome Engagement (ALICE), which employs
a coarse-to-fine strategy to correct low-light aerial degradation.
First, we introduce wavelet transform to design a perturbation
corrector for coarse exposure recovery while preserving details.
Second, inspired by the binocular low-light imaging mechanism
of the human visual system, we introduce uniformly well-exposed
monochrome images to guide a refinement restorer, processing
luminance and chrominance branches separately for further
improved reconstruction. Within this framework, we design a
Reference-based Illumination Fusion Module (RIFM) and an
Illumination Detail Transformation Module (IDTM) for targeted
exposure and detail restoration. Third, we develop a Dual-camera
Low-light Aerial Imaging (DuLAI) dataset to evaluate our proposed ALICE. Extensive qualitative and quantitative experiments
demonstrate the effectiveness of our ALICE, achieving a PSNR
improvement of at least 19.52% over 12 state-of-the-art methods
on the DuLAI Syn-R1440 dataset, while providing more balanced
exposure and richer details.

## Citation
If you found this code useful, please cite the paper. Welcome 👍Fork and Star👍, then I will let you know when we update.

```
@ARTICLE{10948510,
  author={Yuan, Pengwu and Lin, Liqun and Lin, Junhong and Liao, Yipeng and Zhao, Tiesong},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Low-Light Aerial Imaging With Color and Monochrome Cameras}, 
  year={2025},
  volume={63},
  number={},
  pages={1-10},
  keywords={Cameras;Image restoration;Image color analysis;Perturbation methods;Degradation;Lighting;Colored noise;Wavelet transforms;Frequency modulation;Superresolution;Color-monochrome cameras;low-light aerial imaging},
  doi={10.1109/TGRS.2025.3557565}}
```
## Acknowledgements
This work is based on [LDRM](https://github.com/JHLin42in/LDRM).
