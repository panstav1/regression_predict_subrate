
import logging
import os
import pickle
import pandas as pd
from keras.models import model_from_json

log = logging.getLogger(os.path.basename(__file__))

class TrainModel:


    def __init__(self,
                 model_name_file = None,
                 model_weights_file = None,
                 scalerX = None,
                 scalerY = None,
                 num_features = os.environ.get("N_FEATURES", 47),
                 selected_features = os.environ.get("N_FEATURES", [46,22,20,21,24,29,44,6,13,8,45,40,28])):
        """
        @param model_name_file:
        @param model_weights_file:
        @param scalerX:
        @param scalerY:
        """
        if model_weights_file != None:
            self.scalerX = self.load_scaler(scalerX)
            self.scalerY = self.load_scaler(scalerY)
            self.model = self.read_init_model(model_name_file, model_weights_file)
            self.num_features = num_features
            self.selected_features = selected_features
            os.system("rm -rf temp/")
        else:
            self.num_features = num_features
            self.selected_features = selected_features
            self.model = self.read_init_model_no_weights(model_name_file)

    def predict_rate(self, features):
        return self.model.predict(features)

    def def_scaler(self,scalerX,scalerY):
        self.scalerX = scalerX
        self.scalerY = scalerY

    def def_model(self,model):
        self.model = model

    def load_scaler(self, name):
        """

        @param name:
        @return:
        """
        return pickle.load(open('temp/' + name, 'rb'))

    def read_init_model_no_weights(self,model_name_file):
        """
        @param model_name_file:
        @param model_weights_file:
        @return:
        """
        with open('temp/' + model_name_file, 'r') as mdl_name:
            loaded_model_json = mdl_name.read()
            mdl_name.close()
        loaded_model = model_from_json(loaded_model_json)
        return loaded_model

    def read_init_model(self,model_name_file,model_weights_file):
        """

        @param model_name_file:
        @param model_weights_file:
        @return:
        """
        with open('temp/' + model_name_file, 'r') as mdl_name:
            loaded_model_json = mdl_name.read()
            mdl_name.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('temp/' + model_weights_file)
        print("Loaded model from disk")
        return loaded_model

    def read_init_dataset(self, dataset):
        return pd.read_hdf('temp/' + dataset, 'rates')
