# z2x19

Zeit Z2X Face Recognition Demo

## Setup

The basic steps are:

- clone project
- install miniconda
- create conda environment
- run demos

### Clone Project

- `git clone https://github.com/adamhrv/face_workshop`
- you may need to install git first

### Install Miniconda

- Download the install file for your system <https://docs.conda.io/en/latest/miniconda.html>
- open terminal and run `bash the-name-of-miniconda-file.sh`

### Create Conda Environemnt

- open terminal and `cd` into `face_workshop`
- `conda`
- make python virtual environment 
	- you might need install virtualenv `pip3 install --user virtualenv`
	- using virtualenv `virtualenv -p python3 z2x`
	- using python3 `python3 -m venv z2x`
	- using virtualenvwrapper `mkvirtualenv z2x`
	- (going to clarify this later)
- install requirements `pip install -r requirements.txt`

## Demos

- `python demos/python demos/webcam.py`

## Excercise

- Download LFW and use face recognition to find if you're in the dataset, or your doppleganger
- Download LFW: <http://vis-www.cs.umass.edu/lfw/#download>