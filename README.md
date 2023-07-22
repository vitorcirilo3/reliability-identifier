<h1 align="center">Unreliability identifier</h1>

<p align="center">
<img src="https://github.com/vitorcirilo3/underspecification-identifier/blob/main/logo/logo.gif" alt="" data-canonical-src="[https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png](https://github.com/vitorcirilo3/underspecification-identifier/blob/main/logo/logo.gif)" width="771" height="300" />
</p>

<p align="center">🚀 This project aims to identify if a context is unreliable using a methodology based on Item Response Theory.</p>

</br>


```diff
- Source code of the paper "The quest for the reliability of machine learning models in binary classification on tabular data" -
```

***
Summary
=================
<!--ts-->
   * [About](#about)
   * [Prerequisites](#prerequisites)
   * [Quick start](#quick-start)
      * [Pull the docker image](#pull-the-docker-image)
      * [Create the results folder](#create-the-results-folder)
      * [Run docker container](#run-docker-container)
      * [Set up the environment](#set-up-the-environment)
      * [Create the data partitions](#create-the-data-partitions)
      * [Execute the main pipeline](#execute-the-main-pipeline)
<!--te-->

# About
The unreliability identifier is data-centric methodology based on Item Response Theory (IRT) that allows us to identify whether a machine learning context is unreliable. The main objective of our methodology is to create a classifier to identify the reliable related to the stress test of Shifted Performance Evaluation by the use of IRT parameters.

The big picture of our methodology can be seen bellow.

<p align="center">
<img src="https://github.com/vitorcirilo3/underspecification-identifier/blob/main/logo/overview.png" alt="" data-canonical-src="[https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png](https://github.com/vitorcirilo3/underspecification-identifier/blob/main/logo/overview.png)" width="772" height="340" />
</p>

</br>

# Prerequisites
- Docker 20.10.14+

</br>

# Quick start

Follow the steps below to replicate the results that were presented in the paper "The quest for the reliability of machine learning models in binary classification on tabular data"

### Pull the docker image
```
docker pull vitorcirilo3/unrealibility-identifier:1.0
```

### Create the results folder

If you are using ubuntu/debian then just use the command

```
mkdir results
```

### Run docker container

```
docker run -it -v /$(PWD)/results:/root/reliability-identifier/results --name reliability-identifier vitorcirilo3/reliability-identifier:1.0
```

### Set up the environment

```
cd root/reliability-identifier
```

```
git reset --hard
```

```
git pull
```

### Create the data partitions
all set up! Let's start to run the methodology. First step, create the dataset partitions. Execute the python code:
```
python datasets_partition_creation.py
```

### Execute the main pipeline
After the creation of the data partitions, execute the main code to apply the IRT and create the model to identify unreliable contexts
```
main.py
```

PS: this script takes about 6 hours on a computer i7 with 16 GB of ram

## Results
When this script done, all the results will be save at 'results' folder that was mapped outside of the docker


