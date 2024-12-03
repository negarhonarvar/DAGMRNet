# DAGMRNet: Attentive Graph Learning with Dynamic Neighbor Selection for MR Image Reconstruction 
This repository contains a comprehensive model to reconstruct cardiac MRI in k-space from CMRxRecon dataset. In this model we have advanced a graph neural network well-equipped with dynamic attention mechanism which selects neighbors based on their similarity to target node to update the target node relatively. The proposed model is implemented with inspiration from promptMR and DAGL model.

## Architecture
The model implies a cascaded architecture for optimization and effiency. in each cascade a certain set of problems are beign solved which are available under the preliminaries section of the related paper:

<p align="center">
  <img src="https://github.com/user-attachments/assets/e8f62112-b7bb-42f5-a000-65b4afb1d981" width = "400">
</p>

The model levrages PromptUnet as described in PromptMr paper for sensitivity map estimation:

<p align="center">
  <img src="https://github.com/user-attachments/assets/6f493f9b-6d70-4ce6-bbf2-ff8f17428056" width = "400">
</p>

In each cascade, a denoising network is employed to reconstruct the highly-undersampled cardiac MR images, an abstract of each cascade structer is shown bellow:

<p align="center">
   <img src="https://github.com/user-attachments/assets/dc832d13-eef8-403a-8cc8-26d357fe63ab" width = "400">
</p>

