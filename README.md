# CancerPhenotyper

## Setup

Run the following installs to setup the virtual environment using conda.

```bash
conda create --name cancer-phenotyper python=3.8
conda activate cancer-phenotyper
pip3 install torch torchvision
conda install -c conda-forge openslide openslide-python
conda install -c conda-forge numpy scipy pandas matplotlib seaborn scikit-learn scikit-image pillow opencv jupyterlab
```
