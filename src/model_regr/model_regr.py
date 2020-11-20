
import logging
import os
import pickle
from keras.models import model_from_json

log = logging.getLogger(os.path.basename(__file__))

class TrainModel:


    def __init__(self,
                 model_name_file,
                 model_weights_file,
                 scalerX,
                 scalerY,
                 num_features = os.environ.get("N_FEATURES", 47),
                 selected_features = os.environ.get("N_FEATURES", [46,22,20,21,24,29,44,6,13,8,45,40,28])):
        """
        @param model_name_file:
        @param model_weights_file:
        @param scalerX:
        @param scalerY:
        """
        self.scalerX = self.load_scaler(scalerX)
        self.scalerY = self.load_scaler(scalerY)
        self.model = self.read_init_model(model_name_file, model_weights_file)
        self.num_features = num_features
        self.selected_features = selected_features

    def predict_rate(self, features):
        return self.model.predict(features)


    def load_scaler(self, name):
        """

        @param name:
        @return:
        """
        return pickle.load(open('temp/' + name, 'rb'))


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
        os.system("rm -rf temp/")
        print("Loaded model from disk")
        print(loaded_model.summary())
        return loaded_model
