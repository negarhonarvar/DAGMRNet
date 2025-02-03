## Architecture
The model implies a cascaded architecture to solve the reconstruction as an optimization problem, rewritten as an iterative gradient descent method in the k-space domain. In each cascade a certain set of problems are beign solved which together, they solve one iteration of the iterative gradient descent problem.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e8f62112-b7bb-42f5-a000-65b4afb1d981" width = "400">
</p>

The structure of each cascade, implementing the problem sequence in each iteration of iterative gradient descent method is shown below:

<p align="center">
   <img src="https://github.com/user-attachments/assets/dc832d13-eef8-403a-8cc8-26d357fe63ab" width = "400">
</p>

Our proposed model is the Denoising network shown here with "D" symbol.

ACS stands for Automated Calibration Signal which refers to parts of each image in k-space that are fully sampled and are used to estimate the sensitivity maps.
Inverse fourier transform is used to transform k-space signals to image domain.

The model levrages Prompt U-Net as described in PromptMr paper for sensitivity map estimation, which its values are later used in reduce and expand operation for converting k space data to image domain and vice versa. the implementation of the described Prompt U-Net is shown below:

<p align="center">
  <img src="https://github.com/user-attachments/assets/6f493f9b-6d70-4ce6-bbf2-ff8f17428056" width = "400">
</p>



