# Building the environmental framework

## Building the development environment (NVIDIA driver, CUDA, cuDNN)

### Install NVIDIA driver

To view the installable NVIDIA driver versions, run the following command:

```bash
ubuntu-drivers devices # May not be available on WSL2 Ubuntu
```

then install the appropriate driver version:

```bash
sudo apt-get install nvidia-driver-525
```

For a more precise installation of the driver, you can choose to download and install the driver manually ([Web link](https://www.nvidia.com/download/index.aspx)).

If the installation is complete, execute the command to reboot your computer `sudo reboot`. 

Execute the command `nvidia-smi` to see if the driver is successfully installed:

![image-20231007114718531](https://github.com/GithubSherlock/Generating_Trajectories_via_MOT/blob/main/doc/Building%20the%20environmental%20framework.assets/image-20231007114718531.png)

### Install CUDA

[Link to download NVIDIA CUDA driver](https://developer.nvidia.com/cuda-toolkit)

Enter the following commands in a terminal to install the corresponding version of the CUDA and PyTorch drivers in Anaconda's virtual environment. Since detectron2 only supports CUDA 11.3, the CUDA Toolkit version installed here is 11.3.1. (See [detectron2 documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html))

Download and install the driver ([CUDA Toolkit 11.3](https://developer.nvidia.com/cuda-11-3-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local)) manually. Choose "runfile(local)":

``````bash
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
sudo sh cuda_11.3.1_465.19.01_linux.run
``````

Configuring Environment Variables:

```bash
sudo gedit ~/.bashrc
```

The following three paths are added at the end of the text:

```bash
# >>> cuda path >>>
# if you want to change cuda-version, replace the following export sources "cuda-xx.x"
# cuda list: 11.3, 11.7, 12.2
# then use command <source ~/.bashrc> if finised.
export PATH="/usr/local/cuda-11.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.3
# <<< cuda path <<<
```

Save the text after adding it, then use the command `source ~/.bashrc` to read the path, and finally use the command `sudo reboot` to reboot the system.

After rebooting use the command `nvcc -V` to test if cuDNN was successfully installed:

![image-20231007171601397](https://github.com/GithubSherlock/Generating_Trajectories_via_MOT/blob/main/doc/Building%20the%20environmental%20framework.assets/image-20231007171601397.png)

### Install cuDNN

[Link to download NVIDIA cuDNN driver](https://developer.nvidia.com/rdp/cudnn-archive)

Note that in order to download cuDNN you need to register first, after completing and logging in you can download the version of cuDNN that corresponds to the version of CUDA you have installed. Here the cuDNN v8.9.4 for CUDA 11.x is used.

![image-20231007172607569](https://github.com/GithubSherlock/Generating_Trajectories_via_MOT/blob/main/doc/Building%20the%20environmental%20framework.assets/image-20231007172607569.png)

To install a deb package, use the command `dpkg`:

```bash
sudo dpkg -i <path_to_deb_file>
```

Note: Replace "path_to_deb_file" with the path and name of the downloaded deb file.

## Building the programming environment

### Install Anaconda3

First, [download](https://www.anaconda.com/download#downloads) the corresponding installation package, and then install Anaconda3:

```bash
bash ~/Downloads/Anaconda3-2023.09-0-Linux-x86_64.sh
```

Anaconda automatically adds environment variables to PATH, but if the command `conda` is found to be unavailable, environment variables have to be added manually:

```bash
sudo gedit ~/.bashrc
```

then add it at the end of the text:

```bash
export PATH=/home/<yourSysName>/anaconda3/bin:$PATH
```

Note: Replace "yourSysName" with your system's name.

Save and exit and then execute the command `source ~/.bashrc`, finally enter the command `conda list` to test for availability.

### Install PyTorch

First, create and then activate a virtual environment:

```bash
conda create -n <yourEnv> python=<version>
conda activate <yourEnv>
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # Cuda 11.3 is the latest version for detectron2!
```

Note: To use the command `conda`, it is necessary to successfully install Anaconda3 or Miniconda first!

The PyTorch downloads above are for the CPU version only. If you want to use the GPU version of PyTorch, you have to find the download command for the corresponding version in the link.

[Link to download previous versions of PyTorch](https://pytorch.org/get-started/previous-versions/)

My used command:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 # 1.12.1 for PyTorch version, cu113 for CUDA version 11.3
```

### A simple Python code to know if PyTorch with CUDA is available

Run the command `python` in a terminal to activate the Python programming environment.

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

If the development environment is successfully built, the version of PyTorch and CUDA used will be displayed, followed by "True", then you can use your computer's GPU for deep neural networks in programming.

### Clone and install detectron2

[GitHub](https://github.com/facebookresearch/detectron2)

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
```

See the [Documentation](https://detectron2.readthedocs.io/en/latest/index.html) to get tutorials.

### Clone and install Yolov8

[GitHub](https://github.com/ultralytics/ultralytics)

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e .
```

See the [Documentation](https://docs.ultralytics.com/) to get tutorials.

### Install OpenCV

```bash
pip3 install opencv-python
pip3 install opencv-contrib-python
```
