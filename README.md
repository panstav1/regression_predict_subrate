# Prediction of submission rates
 
This repository contains the development of an ML algorithm for predicting the submission rates and holds its API implementation. The API implementation provides the ability of using a pretrained neural network, that is developed in an offline mode, and training a new model with the provided dataset. The initial provided dataset is provided in the code with the files of the pretrained model. This repository comprises a basic component for the functionality of the following:

* Predict submission rates with the pretrained model
* Train the same format of the Neural Network and predict submission rates
   
## Installation and Dependencies
This component is implemented in Python3. Its requirements are specified in the packages.txt in the root folder. In general, a new virtual environment would be beneficial in the installation. Also, it is crucial to advice that linux should be the OS for testing this environment. Regardless of the OS, Docker and Python 3 must be installed.

### Installing from code in bare metal
In this option, a functional mongoDB is essential for the core functionality of the service. To have it up and running from code, please do the following:

```shell
$ git clone https://github.com/panstav1/regression_predict_subrate.git # Clone this repository
$ cd regression_predict_subrate # Go to the downloaded folder
$ python setup.py develop # Install dependencies
$ cd src/model_regr
$ python app.py run # server at http://localhost:4010
```

A server will be running on that session, on port `4010`. You can access it by using `curl`, like in:

```shell
$ curl <host name>:4011/api
``` 



### Docker-based

In this option, a functional mongoDB and docker installed are essential for the core functionality of the service.
With the following code, a docker of the code will be built and run:
```bash
# build Docker container
sudo docker build .

# run Docker container
docker run --rm -d -p 4010:4010 --name regression_predict_subrate
```

### Docker-compose-based (highly recommended)

In this option, a functional mongoDB is included in the docker-compose script. With a simple script, MongoDB and the service is installed:
```bash
# build Docker containers of MongoDB and tng-vnv-dsm
sudo docker-compose build

# run Docker containers
sudo docker-compose up
```


## Developing/Contributing
To contribute to the development, you may use the very same development workflow as for any other Github project. That is, you have to fork the repository and create pull requests.

### Dependencies
In this repository, the following libraries are used (also referenced in the [`packages.txt`](https://github.com/panstav1/regression_predict_subrate/blob/main/packages.txt) file) for development:

* [Numpy](https://numpy.org/) (`v.1.18.5`) - Scientific computing tools with Python
* [Pandas](https://pandas.pydata.org/) (`v.1.0.5`) - Open source data analysis and manipulation tool
* [Scikit-learn](https://scikit-learn.org/stable/) (`v.0.23.1`) - Simple and efficient tools for Machine Learning in Python
* [Flask](https://pypi.org/project/Flask/) (`v.1.1.2`) - A simple framework for building complex web applications.
* [Flask-restplus](https://pypi.org/project/flask-restplus/) (`v.0.13.0 `) - Fully featured framework for fast, easy and documented API development with Flask
* [Tensorflow](https://www.tensorflow.org/) (`v.2.3.1`) - Open source library to develop and train ML models
* [Keras](https://keras.io/) (`v.2.4.3`) - High-level API of TensorFlow 2.0
* [Requests](https://pypi.org/project/requests/) (`v.2.5`) - Python HTTP for Humans 
* [H5py](https://www.h5py.org/) (`v.2.10.0`) - Pythonic interface to the HDF5 binary data format
* [Setuptools](https://pypi.org/project/setuptools/) (`v.50.3.2`) - Easily download, build, install, upgrade, and uninstall Python packages
* [Cython](https://cython.readthedocs.io/en/latest/src/quickstart/install.html) (`v.0.29.21`) - The Cython compiler for writing C extensions for the Python language

Below, the libraries are used for the MongoDB functionalities:

* [Pymongo](https://pymongo.readthedocs.io/en/stable/#) (`v.3.11.1`) - Python distribution containing tools for working with MongoDB

These libraries are installed/updated in the developer's machine when running the command (see above):

```shell
$ python setup.py install (develop)
```



### Submitting/Requesting changes
Changes to the repository can be requested using [this repository's issues](https://github.com/panstav1/regression_predict_subrate/issues) and [pull requests](https://github.com/panstav1/regression_predict_subrate/pulls) mechanisms.


## Usage

The aim of regression_predict_subrate API is the seamless provision of predictions on the submission rates. 
More specifically, it is designed to host a pretrained Neural Network model and a non trained Neural Network model, with
the same structure. The former is available to provide predictions from the very first time of the deployment of the project while the latter
needs to be trained as a first step. The sequential structure of the ML models is defined in Keras below:

* Dense layer of 50 Neurons followed by ReLu activation layer
* Dense layer of 512 layers followed by ReLu activation layer 
* Dense layer of 256 layers followed by ReLu activation layer
* Dense layer of 64 layers followed by ReLu activation layer
* Dense layer of 1 layers followed by Linear activation layer

The structure of the Neural Network is serialized in the JSON format in this [file](https://github.com/panstav1/regression_predict_subrate/blob/main/src/model_regr/binFiles/rate_regr_model.json).
In the same [folder](https://github.com/panstav1/regression_predict_subrate/tree/main/src/model_regr/binFiles), the initial files of the pretrained model and the dataset are located, all
of which will be uploaded in the MongoDB with the deployment of the API.

### Dataset

The dataset is located in the [folder](https://github.com/panstav1/regression_predict_subrate/tree/main/src/model_regr/binFiles/rates_init.h5) in .h5 format for seamless retrieval from the API.
The API has the 

| Features | Meaning |
| -------------------------- | -------- |  
| surveyID | ID of the corresponding service |  
| submissions | # of submissions |  
| views | # of views |  
| v0 | Feature 0 |  
| ... | ... |  
| v46 | Feature 46 | 
| sub_rate | Submission rate, defined as submissions per views |

The dataset is formatted as a DataFrame in order to provide its seamless retrieval from MongoDB directly to the API as soon as it is deployed.

### Pretrained Neural Network Model

The pretrained model is trained in an offline mode under several experimentations with the dataset. The corresponding files that were produced from the offline mode
are located in the [binFiles](https://github.com/panstav1/regression_predict_subrate/tree/main/src/model_regr/binFiles) folder, in which the preprocessing PowerTransformers,
the [model](https://github.com/panstav1/regression_predict_subrate/tree/main/src/model_regr/binFiles/rate_pretrained_model.json) itself and its [weights](https://github.com/panstav1/regression_predict_subrate/tree/main/src/model_regr/binFiles/rate_pretrained_model_wghts.h5)
are located. With the deployment of the API, these are transferred in the MongoDB's GridFS with the same filenames and loaded in the API, ready for use: 

| Action | HTTP Method | Endpoint |  
| -------------------------- | -------- | --------------------------------------- |  
| Get prediction of a single submission rate | `GET` | `curl  -H "Content-Type: application/json" -X GET --data '{"surveyID": 58485, "features": [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,2]}' http://localhost:4010/api/pretrained/predict` |  
| Get predictions of several submission rates | `GET` | `curl  -H "Content-Type: application/json" -X POST --data '[{"surveyID": 58485, "features": [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,2]},{"surveyID": 4234232, "features": [0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,2,0,2]}]' http://localhost:4010/api/pretrained/predict` |

_Note_: It is recommended to use Postman for sending requests, except if you connect with another API.

The data transferred through the endpoint as a JSON object or as a list of JSON objects for the single and the multiple submission rates respectively. 
The validity of the names and the length of the features is performed in the API.

### From Scratch Neural Network Model

The same structure of the pretrained model is provided also for training with the initial dataset. Through the endpoints, specific training optional parameters can be
provided, such as:

`curl -X POST http://localhost:4010/api/new_model/train[?optional_parameter1=X&optional_parameter2=Y&...]`

_Note_: It is recommended to use Postman for sending requests, except if you connect with another API.

with `[]` denoting the position of optional parameters, optional_parameter1 and optional_parameter2 the actual names of the parameters and X,Y their corresponding actual values.
The optional parameters are listed below:

| Optional Training Parameter | Default Value | Format in the Endpoint |  
| -------------------------- | -------- | --------------------------------------- |  
| test_size | `test_size = 0.2` | `http://localhost:4010/api/new_model/train?test_size=0.3` |  
| batch_size | `batch_size = 512` | `http://localhost:4010/api/new_model/train?batch_size=32` |
| epochs | `epochs = 20` | `http://localhost:4010/api/new_model/train?epochs=50` |  
| frac (% of the initial dataset in (0,1]) | `frac = 1` | `http://localhost:4010/api/new_model/train?frac=0.3` |

_Note_: It is recommended to use Postman for sending requests, except if you connect with another API. (for training, it is highly recommended :D )
The combination of the above parameters is feasible through the `&` operator. The API for the prediction works very similar to the API of the pretrained.  

| Action | HTTP Method | Endpoint |  
| -------------------------- | -------- | --------------------------------------- |  
| Get prediction of a single submission rate | `GET` | `curl  -H "Content-Type: application/json" -X GET --data '{"surveyID": 58485, "features": [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,2]}' http://localhost:4010/api/new_model/predict` |  
| Get predictions of several submission rates | `GET` | `curl  -H "Content-Type: application/json" -X POST --data '[{"surveyID": 58485, "features": [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,2]},{"surveyID": 4234232, "features": [0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,2,0,2]}]' http://localhost:4010/api/new_model/predict` |

### Documentation
Generated from swagger, you can log in a browser at:

`http://localhost:4010/api`

Documentation is provided with automatic interactive API documentation.

### Other Endpoints
#### Health endpoint
A useful endpoint is the one that checks that the docker is up:

`curl -X GET http://localhost:4010/api/health`

#### Logger endpoint
The endpoint to fetch the internal logger file is:
`curl -X GET http://localhost:4010/api/log`


## Useful Links  
  
To support working and testing with the regression_predict_subrate, it is optional (yet, highly recommendable)) to use next tools:  
  
* [Robomongo](https://robomongo.org/download) - Robomongo 0.9.0-RC4  
* [POSTMAN](https://www.getpostman.com/) - Chrome Plugin for HTTP communication  

---  
## Lead Developers  
  
The following lead developers are responsible for this repository and have admin rights. They can, for example, merge pull requests.  
  
* Panagiotis Stavrianos (panstav1)  

Please use the GitHub issues and the e-mail for feedback.