
import logging, os, functools, time
from logging.handlers import RotatingFileHandler
from flask import Flask, Blueprint, request, Response
from model_regr.inouttransforms import InOutTransforms as iotran
from flask_restplus import Api, Resource
from model_regr.db_models.models import MongoDB as mongo
from model_regr.model_regr import TrainModel as rec_mod
from model_regr.logger import PlainnLogger


Trn_mod = mongo(collection='trained_model')

mdl_name = os.environ.get("MODEL_NAME", "rate_regr_model.json")
mdl_weights = os.environ.get("MODEL_WEIGHTS", "rate_regr_model_wghts.h5")
scalerX = os.environ.get("MODEL_NAME", "scalerX.pkl")
scalerY = os.environ.get("MODEL_WEIGHTS", "scalerY.pkl")

mdl_name = Trn_mod.retr_init_file(mdl_name)
mdl_weights = Trn_mod.retr_init_file(mdl_weights)

scalerX = Trn_mod.retr_init_file(scalerX)
scalerY = Trn_mod.retr_init_file(scalerY)


rec_mod = rec_mod(mdl_name, mdl_weights,scalerX,scalerY)
ioObj = iotran()


app = Flask(__name__)
blueprint = Blueprint('api', __name__, url_prefix='/api')
api = Api(blueprint, version="0.1",
          title='Pre-trained Regression Model Flask API on submission rates',
          description="Pre-trained Regression Model Flask API on submission rates")


app.register_blueprint(blueprint)

LOG = PlainnLogger.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

def catch_exception(f):
    @functools.wraps(f)
    def func(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except ValueError as e:
            print('Caught an exception in',f.__name__)

    return func

@api.route('/log', methods = ['GET'])
class RegLocalLog(Resource):
    def get(self):
        with open("logger","r") as f:
            return Response(f.read(),mimetype='text/plain')

@api.route('/health', methods =['GET'])
class RegHealth(Resource):

    def get(self):
        return ioObj.generic_resp(200,'application/json',
                                  ioObj.json_d(ioObj.success_message('OK check')))

@catch_exception
@api.route('/predict', methods =['POST'])
class RegPredict(Resource):

    def post(self):

        start_time = time.time()

        try:
            data_load = ioObj.json_l(request.data)
            data_req = ioObj.conv_json_to_df(data_load)
        except Exception as e:
            return ioObj.generic_err400_with_resp('Wrong data input format', start_time)

        if data_req.shape[1] != 2 or any(x not in data_req.columns for x in ['features','surveyID']):
            return ioObj.generic_err400_with_resp("Field 'features' and/or 'surveyID' are not provided correctly.", start_time)


        if not all(data_req['features'].apply(len) == rec_mod.num_features):
            return ioObj.generic_err400_with_resp('Not correct type or length of features input. Insert {feat} integers/floats in the array'
                                                  .format(feat = rec_mod.num_features), start_time)




        data_req['selected_features'] = data_req['features'].apply(lambda x: [x[idx] for idx in rec_mod.selected_features])

        scld_features = rec_mod.scalerX.transform(data_req['selected_features'].to_list())
        predicted_rate = rec_mod.predict_rate(scld_features)
        result = rec_mod.scalerY.inverse_transform(predicted_rate)

        data_req['pred_rate'] = result
        LOG.info("Predicted submission rate\n")

        results = data_req[['surveyID', 'pred_rate']].to_dict('records')

        LOG.info("Prediction of submission rate " + str(ioObj.json_d(results)),
                 extra={"status": 200, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})
        return ioObj.generic_resp(200, 'application/json',
                                  ioObj.json_d(ioObj.success_message(results)))


class MakeFileHandler(RotatingFileHandler):
    def __init__(self, filename, maxBytes, backupCount, mode='a', encoding=None, delay=0):
        RotatingFileHandler.__init__(self, filename,maxBytes, backupCount )


if __name__ == '__main__':
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    handler = MakeFileHandler('logger', maxBytes=1000, backupCount=1)
    handler.setLevel(logging.DEBUG)
    app.logger.addHandler(handler)
    app.run(host='0.0.0.0', port=os.getenv('PORT', 4010), debug=True)
