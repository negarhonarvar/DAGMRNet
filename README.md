# DAGMRNet: Attentive Graph Learning with Dynamic Neighbor Selection for MR Image Reconstruction 
This repository contains a comprehensive model to reconstruct cardiac MRI in k-space from CMRxRecon dataset. In this model we have advanced a graph neural network well-equipped with dynamic attention mechanism which selects neighbors based on their similarity to target node to update the target node relatively. The proposed model is implemented with inspiration from promptMR and DAGL model.

## Architecture
The model implies a cascaded architecture for optimization and effiency.
<p align="center">
  <img src="https://github.com/user-attachments/assets/e8f62112-b7bb-42f5-a000-65b4afb1d981" width = "400">
</p>
