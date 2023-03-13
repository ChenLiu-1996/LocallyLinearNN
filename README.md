# Improving Continuity of Neural Networks

Chen Liu (chen.liu.cl2482@yale.edu)



## Environement Setup
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
conda create --name $OUR_CONDA_ENV pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 numpy -c pytorch -c anaconda
conda activate $OUR_CONDA_ENV
conda install -c anaconda scikit-image pillow matplotlib seaborn tqdm
```
