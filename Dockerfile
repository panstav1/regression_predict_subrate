
FROM python:3.6-slim

ENV MONGO_URL mongo
ENV MONGO_PORT 27017

# Install basics
RUN apt-get update && apt-get install -y git  # We net git to install other tng-* tools.
RUN apt-get install -y python3-pip
RUN pip install flake8
RUN pip install setuptools \
		numpy

#
#
WORKDIR /
ADD . /regression_predict_subrate
WORKDIR /regression_predict_subrate
RUN python setup.py develop

#
# Runtime
#
WORKDIR /regression_predict_subrate/src/model_regr
#RUN pwd
EXPOSE 4010

CMD ["python", "/regression_predict_subrate/src/model_regr/app.py"]
