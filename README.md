# DAGMRNet: Dynamic Attentive Graph Learning Cardiac MR Image Reconstruction 

This repository contains the pytorch implementation of DAGMRNet, a comprehensive model to reconstruct multi coil cardiac MRI in k-space from CMRxRecon 2024 dataset. 

## Method

This model utilizes a " Dynamic Attentive Graph Learning " model as a denoising network for reconstructing cardiac MRI based on " Self Similarity " Image prior. The Architecture of our proposed model is shown below:

<p align = "center">
   <img src="https://github.com/user-attachments/assets/30f8b733-4845-4fbe-b42b-ede9e59f98c4" width = "400" >
</p>

Check the Readme.md of [model](https://github.com/negarhonarvar/DAGMRNet/tree/main/models) Directory for more details.

## Prerequisites 📋

Required libraries and dependencies are listed as a code block inside the requirements.txt file. run the code below and install them:

    pip install -r requirements.txt


## Dataset :anatomical_heart:

This model is trained on Training Set of Multi Coil Cine accelerated cardiac MRI's of CMRxRecon Dataset and evaluated on its Validation Set datas, which are intended for CMR reconstruction evaluation. Check the [Link](https://cmrxrecon.github.io/2024/Home.html)
 and request for the dataset.

## Results :bar_chart:

## Table 3-4: Training Results of the Proposed Model and PromptMR for LVOT, LAX, and SAX

| Model       | PSNR/SSIM          | Number of Fully Sampled Signals | Acceleration Factor | Number of Variables | Number of Cascades |
|------------|--------------------|---------------------------------|---------------------|----------------------|--------------------|
| **PromptMR** | 38.28 / 0.9560      | 16                              | 4x, 8x, 10×                 | 90 M               | 12                 |
| **Proposed Model** | 37.10 / 0.9510  | 16                              | 4x, 8x, 10×              | 16.1 M               | 4                  |


## Training/Inference Codes & Pretrained models :brain:

Current weights of model are accessible in [Best Weights Directory](https://github.com/negarhonarvar/DAGMRNet/tree/main/Best_Weights) Directory of this repository.
Set the variable

    args.mode == "test"
    
and enjoy reconstruction your CMR images!

