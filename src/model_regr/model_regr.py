# Libraries
import logging
import os
import pickle
from keras.models import model_from_json
from keras import backend

log = logging.getLogger(os.path.basename(__file__))


class TrainModel:
    """
    TrainModel Class
    Class instance of Regressors
    """
    def __init__(self,
                 model_name_file=None,
                 model_weights_file=None,
                 scalerX=None,
                 scalerY=None,
                 num_features=os.environ.get("N_FEATURES", 47),
                 check_new=False):
        """
        @param model_name_file: Object, model file
        @param model_weights_file: Object, File of the weights of the model
        @param scalerX: Object, scaler of the features
        @param scalerY: Object, scaler of the dependent value
        @param check_new: Boolean, If is new model
        """
        if not check_new:
            self.scalerX = self.load_scaler(scalerX)
            self.scalerY = self.load_scaler(scalerY)
            self.model = self.read_init_model(model_name_file, model_weights_file)
            self.num_features = num_features
            os.system("rm -rf temp_fromfs/")
        else:
            self.num_features = num_features
            self.model = self.read_init_model_no_weights(model_name_file)
            if scalerX is not None:
                self.scalerX = self.load_scaler(scalerX)
                self.scalerY = self.load_scaler(scalerY)

    def predict_rate(self, features):
        """
        Function for predicting rate from the model
        @param features: List, features to get prediction
        @return: Array of predictions
        """
        return self.model.predict(features)

    def def_scaler(self, scalerX, scalerY):
        """
        Setter of scalers in the object
        @param scalerX: Object, scaler of independent values
        @param scalerY: Object, scaler of dependent values
        """
        self.scalerX = scalerX
        self.scalerY = scalerY

    def def_model(self, model):
        """
        Setter of model
        """
        self.model = model

    def load_scaler(self, name):
        """
        Function of loading the scalers
        @param name: string, file of the scaler
        @return: Object
        """
        return pickle.load(open('temp_fromfs/' + name, 'rb'))

    def coeff_determination(self, y_true, y_pred):
        """
        R2 score for the results of the model fit
        @param y_true: Array, Groundtruth dependent values
        @param y_pred: Array, Predictions of the model
        @return: Float, R2 score
        """
        SS_res = backend.sum(backend.square(y_true - y_pred))
        SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
        return 1 - SS_res / (SS_tot + backend.epsilon())

    def read_init_model_no_weights(self, model_name_file):
        """
        Function of reading new models with no weights
        @param model_name_file: String, filename of the new model
        @return: Object, model
        """
        with open('temp_fromfs/' + model_name_file, 'r') as mdl_name:
            loaded_model_json = mdl_name.read()
            mdl_name.close()
        loaded_model = model_from_json(loaded_model_json)
        return loaded_model

    def read_init_model(self, model_name_file, model_weights_file):
        """
        Function of reading new models and weights
        @param model_name_file: String, filename of the model
        @param model_weights_file: String, filename of the weights
        @return: Object, model with weights
        """
        with open('temp_fromfs/' + model_name_file, 'r') as mdl_name:
            loaded_model_json = mdl_name.read()
            mdl_name.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('temp_fromfs/' + model_weights_file)
        return loaded_model
