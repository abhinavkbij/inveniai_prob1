# inveniai_prob1
Assignment for ML role at inveniai, this repo includes artifacts for problem1


## folder structure
- notebooks folder contains all the notebooks
 - tensorflow_dl.ipynb trains a simple deep neural network with embedding layer
 - test.ipynb includes all the tests and trials with postgresql, rabbitmq and also the code for training model using scikit-learn's alogrithms like naive-bayes, randomforest and gradient boosting with suppport for pipeline(this is model used in inference api)
 - tf_new.ipynb includes trial for bert model from tf hub
**Note**: All the notebooks were used with custom kernels/virtual environments that needs to be setup before running
- rabbitmq
 - this folder includes code for setting up rabbitmq on debian and also running the consumer and producer to send data for inference
- src
 - this folder contains files like Dockerfile, trained model pickles, api file etc.

## Running the api
- first build the docker image and run the container
- go to rabbitmq folder and run `send.py` and  `receive.py` in separate shells
- you might need to resolve some directory related errors when running first time

## Flow of program
- rabbitmq producer sends message for inference
- rabbitmq consumer consumes the message and makes an api call to the inference service
- inference service stores the predictions in sqlite db and also return the inference in response

**Note**: AWS EC2 instance with debian 12 was used for creating/testing the above code, depending upon your platform/python version, you might have to make some changes