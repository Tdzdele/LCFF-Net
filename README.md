# [LCFF-Net: a lightweight cross-scale feature fusion network for tiny target detection in UAV aerial imagery](https://doi.org/10.1371/journal.pone.0315267)

Official PyTorch implementation of LCFF-Net.

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
In the field of UAV aerial image processing, ensuring accurate detection of tiny targets is essential. Current UAV aerial image target detection algorithms face challenges such as low computational demands, high accuracy, and fast detection speeds. To address these issues, we propose an improved, lightweight algorithm: LCFF-Net. First, we propose the LFERELAN module, designed to enhance the extraction of tiny target features and optimize the use of computational resources. Second, a lightweight cross-scale feature pyramid network (LC-FPN) is employed to further enrich feature information, integrate multi-level feature maps, and provide more comprehensive semantic information. Finally, to increase model training speed and achieve greater efficiency, we propose a lightweight, detail-enhanced, shared convolution detection head (LDSCD-Head) to optimize the original detection head. Moreover, we present different scale versions of the LCFF-Net algorithm to suit various deployment environments. Empirical assessments conducted on the VisDrone dataset validate the efficacy of the algorithm proposed. Compared to the baseline-s model, the LCFF-Net-n model outperforms baseline-s by achieving a 2.8% increase in the mAP50 metric and a 3.9% improvement in the mAP50−95 metric, while reducing parameters by 89.7%, FLOPs by 50.5%, and computation delay by 24.7%. Thus, LCFF-Net offers high accuracy and fast detection speeds for tiny target detection in UAV aerial images, providing an effective lightweight solution.
</details>

## Installation

```
pip install -e .
```

## Training

```
python Train.py
```

## Acknowledgement

The code base is built with [Ultralytics](https://github.com/ultralytics/ultralytics).

We sincerely thank all those who contributed to this study.

## Citation

```
@article{10.1371/journal.pone.0315267,
    doi = {10.1371/journal.pone.0315267},
    author = {Tang, Daoze AND Tang, Shuyun AND Fan, Zhipeng},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {LCFF-Net: A lightweight cross-scale feature fusion network for tiny target detection in UAV aerial imagery},
    year = {2024},
    month = {12},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pone.0315267},
    pages = {1-24},
    number = {12},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tdzdele/LCFF-Net&type=Date)](https://www.star-history.com/#Tdzdele/LCFF-Net&Date)
