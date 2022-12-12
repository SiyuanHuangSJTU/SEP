In order to carry out the task of graph classification, a new conda environment can be constructed using the following code.


```
conda create -n SEP python==3.7.11
conda activate SEP

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.8.0+cu111.html

pip install torch-sparse==0.6.11 -f https://data.pyg.org/whl/torch-1.8.0+cu111.html

pip install torch-geometric==2.0.1
```
