# Computer Vision Face Workshop

## Setup

The basic steps are:

- clone project
- install miniconda
- create conda environment
- run demos

### Clone Project

- `git clone https://github.com/adamhrv/face_workshop`
- you may need to [install git first](https://gist.github.com/derhuerst/1b15ff4652a867391f03)

### Install Miniconda

- Download the install file for your system <https://docs.conda.io/en/latest/miniconda.html>
- open terminal and run `bash the-name-of-miniconda-file.sh`

### Create Conda Environemnt

- open terminal and `cd` into `face_workshop`

On Linux:

```
conda env create -f environment.yml
conda activate face_workshop
```

On MacOS:

```
conda create -n face_workshop pytho=3.7
conda activate face_workshop
pip install -r requirements.txt
conda install nb_conda
conda install -n face_workshop nb_conda_kernels 
```

Building dlib may take a while. Maybe install or upgrade gcc. If issues, check <https://github.com/davisking/dlib/>

Windows:

TBD
