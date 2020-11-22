# Libraries
import numpy as np
import logging, os, functools, time
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from logging.handlers import RotatingFileHandler
from flask import Flask, Blueprint, request, Response
from flask_restplus import Api, Resource
from model_regr.inouttransforms import InOutTransforms as iotran
from model_regr.db_models.models import MongoDB as mongo
from model_regr.model_regr import TrainModel as rec_mod
from model_regr.logger import PlainnLogger
import pickle

# Initialize names of pretrained files
mdl_pretrained_name = os.environ.get("MODEL_NAME", "rate_pretrained_model.json")
mdl_pretrained_weights = os.environ.get("MODEL_WEIGHTS", "rate_pretrained_model_wghts.h5")
scaler_pretrained_X = os.environ.get("SCALER_X", "scaler_pretrained_X.pkl")
scaler_pretrained_Y = os.environ.get("SCALER_Y", "scaler_pretrained_Y.pkl")

# Initialize names of new model
mdl_new_name = "rate_new_model.json"
mdl_new_weights_filename = "rate_new_model_wghts.h5"
scaler_new_X_filename = 'scaler_new_X.pkl'
scaler_new_Y_filename = 'scaler_new_Y.pkl'

# Init db from pretrained model
Trn_mod = mongo(collection='trained_model')

# Retrieve init files of pretrained models
mdl_pretrained_name = Trn_mod.retr_init_file(mdl_pretrained_name)
mdl_pretrained_weights = Trn_mod.retr_init_file(mdl_pretrained_weights)
scaler_pretrained_X = Trn_mod.retr_init_file(scaler_pretrained_X)
scaler_pretrained_Y = Trn_mod.retr_init_file(scaler_pretrained_Y)

# Retrieve init files of new model (if trained)
mdl_new_weights = Trn_mod.retr_init_file(mdl_new_weights_filename)
scaler_new_X = Trn_mod.retr_init_file(scaler_new_X_filename)
scaler_new_Y = Trn_mod.retr_init_file(scaler_new_Y_filename)

# Retrieve new model if trained before.
# Else, fetch the format of the pretrained without weights
if Trn_mod.retr_init_file(mdl_new_name) is None:
    new_mdl_name = Trn_mod.retr_init_file(mdl_pretrained_name)
else:
    new_mdl_name = Trn_mod.retr_init_file(mdl_new_name)

# Retrieve dataset, read it and remove any np.Inf
# from the dependent variable
init_dataset = Trn_mod.retr_init_file('rates_init.h5')
init_dataset = Trn_mod.read_init_dataset('rates_init.h5')
init_dataset = init_dataset[init_dataset['sub_rate'] != np.Inf]

# Define early stopping from val_loss
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Define db for the new model
new_model_db = mongo(collection='new_model', init=False)

# Init of the two models
model_new = rec_mod(new_mdl_name, mdl_new_weights, scaler_new_X, scaler_new_Y, check_new=True)
init_mod = rec_mod(mdl_pretrained_name, mdl_pretrained_weights, scaler_pretrained_X, scaler_pretrained_Y)

# Init of the InputOutput Transformation object
ioObj = iotran()

# Define of the Flask API and the model
app = Flask(__name__)
blueprint = Blueprint('api', __name__, url_prefix='/api')
api = Api(blueprint, version="0.1",
          title='Pre-trained and New Regression Model Flask API on submission rates',
          description="Pre-trained and New Regression Model Flask API on submission rates")

# Register the api url blueprint
app.register_blueprint(blueprint)

# Definition of logger
LOG = PlainnLogger.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def catch_exception(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            print('Caught an exception in', f.__name__)

    return func


# Endpoint for fetching the local file logger
@api.route('/log', methods=['GET'])
class RegLocalLog(Resource):
    def get(self):
        with open("logger", "r") as f:
            return Response(f.read(), mimetype='text/plain')


# Endpoint for fetching the health
@api.route('/health', methods=['GET'])
class RegHealth(Resource):

    def get(self):
        return ioObj.generic_resp(200, 'application/json',
                                  ioObj.json_d(ioObj.success_message('OK check')))


@catch_exception
@api.route('/pretrained/predict', methods=['GET'])
class RegPretrainedPredict(Resource):
    """
    RegPretrainedPredict Class
    Class instance for predicting with the pretrained model
    """

    def get(self):
        """
        GET HTTP call
        @return: 400 code with info or 200 with results
        """
        # Start counting time
        start_time = time.time()

        # Check format of the input JSON. Return 400 if false format
        try:
            data_load = ioObj.json_l(request.data)
            data_req = ioObj.conv_json_to_df(data_load)
        except Exception as e:
            return ioObj.generic_err400_with_resp('Wrong data input format', start_time)

        #
        # Check if names/keys of the JSON are correct. Return 400 if false format
        if data_req.shape[1] != 2 or any(x not in data_req.columns for x in ['features', 'surveyID']):
            return ioObj.generic_err400_with_resp("Field 'features' and/or 'surveyID' are not provided correctly.",
                                                  start_time)
        #
        # Check format and the # of the features. Return 400 if false format
        if not all(data_req['features'].apply(len) == init_mod.num_features):
            return ioObj.generic_err400_with_resp(
                'Not correct type or length of features input. Insert {feat} integers/floats in the array'
                    .format(feat=init_mod.num_features), start_time)

        # Transform data with the pretrained scaler
        # Predict Submission Rate and inverse transform of the result
        scld_features = init_mod.scalerX.transform(data_req['features'].to_list())
        LOG.info("Transformed feature scaling",
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})
        predicted_rate = init_mod.predict_rate(scld_features)
        LOG.info("Predict submission rate",
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})
        result = init_mod.scalerY.inverse_transform(predicted_rate)

        # Make result part of dataframe
        data_req['pred_rate'] = result
        LOG.info("Predicted submission rate")

        # Transform surveyID and features into JSON
        results = data_req[['surveyID', 'pred_rate']].to_dict('records')

        # Return OK code and results into JSON
        LOG.info("Prediction of submission rate " + str(ioObj.json_d(results)),
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})
        return ioObj.generic_resp(200, 'application/json',
                                  ioObj.json_d(ioObj.success_message(results)))


@catch_exception
@api.route('/new_model/train', methods=['POST'])
class RegNewTrain(Resource):
    """
    RegNewTrain Class
    Class instance for training the new model
    """

    def post(self, init_dataset=init_dataset):
        """
        POST HTTP request
        @param init_dataset: Init dataset loaded
        @return: 200 code with history of the fit
        """
        start_time = time.time()

        # Init the main parameters of training
        test_size = request.args.get('test_size', default=0.2, type=float)
        batch_size = request.args.get('batch_size', default=512, type=int)
        epochs = request.args.get('epochs', default=20, type=int)
        frac = request.args.get('frac', default=1, type=float)

        if not (type(frac) is float and (1 >= frac >= 0.0001)):
            return ioObj.generic_err400_with_resp(
                'Wrong format of frac feature. Please, provide a float number in (0,1]', start_time)
        LOG.info("Loaded formatted data",
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})

        # Fetch frac of the dataset
        frac_dataset = init_dataset.sample(frac=frac)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(frac_dataset.iloc[:, 3:-1],
                                                            frac_dataset.iloc[:, -1],
                                                            test_size=test_size)

        LOG.info("Train test split",
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})

        # Init of two PowerTransformer objects
        model_new.def_scaler(PowerTransformer(), PowerTransformer())

        # Fit on X_train and transform both test and train samples
        X_train = model_new.scalerX.fit_transform(X_train)
        X_test = model_new.scalerX.transform(X_test)

        # Fit on y_train and transform both test and train samples
        y_train = model_new.scalerY.fit_transform(y_train.to_numpy().reshape(-1, 1))
        y_test = model_new.scalerY.transform(y_test.to_numpy().reshape(-1, 1))

        LOG.info("Transformed input and output data",
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})

        # Set the model regressor and compile
        regressor = model_new.model
        regressor.compile(optimizer='adam', loss='mse', metrics=['mae', model_new.coeff_determination])

        LOG.info("Compiled the new model and about to fit it",
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})

        # Fit regressor
        history = regressor.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            validation_data=(X_test, y_test)
        )

        LOG.info("Fitted model. Output the results",
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})

        # Format the results
        final_results = {}
        for key in history.history.keys():
            final_results[key] = history.history[key]

        # Serialize format of model, weights, and PowerTransformers
        with open(mdl_new_name, "w") as json_file:
            json_file.write(regressor.to_json())
        regressor.save_weights(mdl_new_weights_filename)
        pickle.dump(model_new.scalerX, open(scaler_new_X_filename, 'wb'))
        pickle.dump(model_new.scalerY, open(scaler_new_Y_filename, 'wb'))

        # Load files into MongoDB
        new_model_db.load_local_files()

        LOG.info("Loaded files of model, weights and PowerTransformers. Outputting the results",
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})
        return ioObj.generic_resp(200, 'application/json',
                                  ioObj.json_d(ioObj.success_message(final_results)))


@catch_exception
@api.route('/new_model/predict', methods=['GET'])
class RegNewPredict(Resource):
    """
    RegNewPredict Class
    Class instance for endpoint of predicting with the new model
    """

    def get(self):
        """
        Get HTTP call
        @return: 400 code with information
        @return: 200 with prediction in format of JSON
        """
        # Start counting
        start_time = time.time()

        # Fetch the new model from the database. If not present, the model is not trained and aborting
        fetch = new_model_db.fs.find_one({'filename': new_mdl_name})
        if fetch is None:
            LOG.info("New model is not trained",
                     extra={"status": 404, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})
            return ioObj.generic_resp(404, 'application/json',
                                      ioObj.json_d(ioObj.error_message('New model is not trained')))

        #
        # Check format of the input JSON. Return 400 if false format
        try:
            data_load = ioObj.json_l(request.data)
            data_req = ioObj.conv_json_to_df(data_load)
        except Exception as e:
            return ioObj.generic_err400_with_resp('Wrong data input format', start_time)

        #
        # Check if names/keys of the JSON are correct. Return 400 if false format
        if data_req.shape[1] != 2 or any(x not in data_req.columns for x in ['features', 'surveyID']):
            return ioObj.generic_err400_with_resp("Field 'features' and/or 'surveyID' are not provided correctly.",
                                                  start_time)

        # Check format and the # of the features. Return 400 if false format
        if not all(data_req['features'].apply(len) == model_new.num_features):
            return ioObj.generic_err400_with_resp(
                'Not correct type or length of features input. Insert {feat} integers/floats in the array'
                    .format(feat=init_mod.num_features), start_time)

        # Transform data with the new trained scaler
        # Predict Submission Rate and inverse transform of the result
        scld_features = model_new.scalerX.transform(data_req['features'].to_list())
        LOG.info("Transformed feature scaling",
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})
        predicted_rate = model_new.predict_rate(scld_features)
        LOG.info("Predict submission rate",
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})
        result = model_new.scalerY.inverse_transform(predicted_rate)

        # Import results into pandas
        data_req['pred_rate'] = result

        # Output results with surveyID in JSON format
        results = data_req[['surveyID', 'pred_rate']].to_dict('records')

        # Output the results with 200
        LOG.info("Prediction of submission rate " + str(ioObj.json_d(results)),
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})
        return ioObj.generic_resp(200, 'application/json',
                                  ioObj.json_d(ioObj.success_message(results)))


class MakeFileHandler(RotatingFileHandler):
    def __init__(self, filename, maxBytes, backupCount, mode='a', encoding=None, delay=0):
        RotatingFileHandler.__init__(self, filename, maxBytes, backupCount)


# Init of the app and the logger
if __name__ == '__main__':
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    handler = MakeFileHandler('logger', maxBytes=1000, backupCount=1)
    handler.setLevel(logging.DEBUG)
    app.logger.addHandler(handler)
    app.run(host='0.0.0.0', port=os.getenv('PORT', 4010), debug=True)
