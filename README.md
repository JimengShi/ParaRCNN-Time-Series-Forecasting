# ParaRCNN-Time-Series-Forecasting
Official codes for "[Explainable Parallel RCNN with Novel Feature Representation for Time Series Forecasting](https://arxiv.org/abs/2305.04876)" accepted by [AALTD workshop at ECML-PKDD 2023](https://ecml-aaltd.github.io/aaltd2023/).

<strong>Abstract</strong>. Accurate time series forecasting is a fundamental challenge in data science, as it is often affected by external covariates such as weather
or human intervention, which in many applications, may be predicted with reasonable accuracy. We refer to them as predicted future covariates.
However, existing methods that attempt to predict time series in an iterative manner with auto-regressive models end up with exponential
error accumulations. Other strategies that consider the past and future in the encoder and decoder respectively limit themselves by dealing with
the past and future data separately. To address these limitations, a novel feature representation strategy - shifting - is proposed to fuse the past
data and future covariates such that their interactions can be considered. To extract complex dynamics in time series, we develop a parallel deep
learning framework composed of RNN and CNN, both of which are used hierarchically. We also utilize the skip connection technique
to improve the model’s performance. Extensive experiments on three datasets reveal the effectiveness of our method. Finally, we demonstrate
the model interpretability using the Grad-CAM algorithm.

<div align="left">
<img src="https://github.com/JimengShi/ParaRCNN-Time-Series-Forecastinga/blob/master/images/model_framework.png" alt="model_framework.png" >
</div>

## Environment Deployment
- conda create -n `EVN_NAME` python=3.8
- pip install -r requirement.txt

## Running codes
- method 1: cd model, then python `FILE_NAME`.py
- method 2: directly go to the `notebooks` folder and run `.ipynb` files.

## Citation
BibTeX

@article{shi2023explainable,

  title={Explainable Parallel RCNN with Novel Feature Representation for Time Series Forecasting},
  
  author={Shi, Jimeng and Myana, Rukmangadh and Stebliankin, Vitalii and Shirali, Azam and Narasimhan, Giri},
  
  journal={arXiv preprint arXiv:2305.04876},
  
  year={2023}
}

[Citation of other formats](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C10&q=Explainable+Parallel+RCNN+with+Novel+Feature+Representation+for+Time+Series+Forecasting&btnG=#d=gs_cit&t=1690224560627&u=%2Fscholar%3Fq%3Dinfo%3AcVxubsFTkIYJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)
