<h1 align="center">Unreliability identifier</h1>

<p align="center">
<img src="https://github.com/vitorcirilo3/underspecification-identifier/blob/main/logo/logo.gif" alt="" data-canonical-src="[https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png](https://github.com/vitorcirilo3/underspecification-identifier/blob/main/logo/logo.gif)" width="771" height="300" />
</p>

<p align="center">🚀 This project aims to identify if a context is unreliable using a methodology based on Item Response Theory.</p>

</br>


```diff
- The source code will be released once the article referring to the work is accepted -
```

***
Summary
=================
<!--ts-->
   * [About](#about)
   * [Prerequisites](#prerequisites)
   * [How to use](#how-to-use)
      * [Pull the docker image](#pull-the-docker-image)
      * [Creation of classifier model](#creation-of-classifier-model)
      * [Classify a machine learning context](#classify-a-machine-learning-context)
   * [Technologies](#technologies)
<!--te-->


# About
The unreliability identifier is data-centric methodology based on Item Response Theory (IRT) that allows us to identify whether a machine learning context is unreliable. The main objective of our methodology is to create a classifier to identify the reliable related to the stress test of Shifted Performance Evaluation by the use of IRT parameters.

The big picture of our methodology can be seen bellow.

<p align="center">
<img src="https://github.com/vitorcirilo3/underspecification-identifier/blob/main/logo/overview.png" alt="" data-canonical-src="[https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png](https://github.com/vitorcirilo3/underspecification-identifier/blob/main/logo/overview.png)" width="772" height="340" />
</p>


# Prerequisites
- Docker 20.10.14+

# Quick start

The step-by-step to execute the Unreliability identifier will be describe bellow:

## Pull the docker image
```
docker pull vitorcirilo3/unrealibility-identifier:1.0
```

## Create a folder called "results"

If you are using ubuntu/debian then just use the command

```
mkdir results
```

## Run docker container

```
docker run -it -v /$(PWD)/results:/root/reliability-identifier/results --name reliability-identifier vitorcirilo3/reliability-identifier:1.0
```

## Go to root folder

```
cd root
```

## Let's clone this repository to there!
```
git clone https://github.com/vitorcirilo3/reliability-identifier.git
```


## Creation of classifier model

This command will create the classifier from the our methodology. By default we are using 128 data partitions, however, it is very easy to change that. You need only the input-list-datasets file. This makes it very easy to perform new tests and update the model if necessary.

```
docker run -d -it --rm $(pwd):/src vitorcirilo3/underspecification-identifier:1.0 /src/main.sh
```

This command will execute our methodology with 128 data partitions from 21 datasets. When the execution be done we will have a model capable to identify unreliability. This model will be in the pkl format and if would like to verify if a new context if unreliability we only need to use the following command 

## Classify a machine learning context
```
docker run -d -it --rm $(pwd):/src vitorcirilo3/underspecification-identifier:1.0 /src/classify.sh dataset_input_name
```

When the execution done it will show at screen "The machine learning context is normal" or "Attention! The machine learning context is unreliable!"


# Technologies
 - Docker 20.10.14
 - Python 3.8.8
 - R 3.5.0
 - Cookiecutter 1.0
