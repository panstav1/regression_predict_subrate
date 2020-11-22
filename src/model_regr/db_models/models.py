# Libraries
import io, glob as g
import pandas as pd
import pymongo, gridfs, functools
import os
import errno


#
def catch_exception(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except TypeError as e:
            print('Caught an exception in', f.__name__)
        return func


class MongoDB:
    """
    DB class
    Class instance is a collection to perform operations on
    """

    # Default settings of MongoDB
    def_host = "localhost"
    def_port = "27017"
    def_database = "rates"

    MONGO_URI = 'mongodb://{0}:{1}'
    ID = '_id'
    INC = '$inc'
    SET = '$set'
    dict_users = 'dict_users'


    def __init__(self, collection, init=True):
        """
        Init function of the MongoDB Object
        @param collection: String, Name of the collection
        @param init: Boolean, if load files
        """
        self.host = os.environ.get("MONGO_URL", self.def_host)
        self.port = os.environ.get("MONGO_PORT", self.def_port)
        self.database = self.def_database
        self.conn = pymongo.MongoClient(self.MONGO_URI.format(self.host, self.port))
        self.collection = self.conn[self.database][collection]
        self.fs = gridfs.GridFS(self.conn[self.database])
        self.db = self.conn[self.database]
        if init:
            self.load_init_files()

    def load_init_files(self):
        """
        Load initial files in the binFiles folder for the pretrained model
        """
        listOfFiles = []
        for filetype in ['*.h5', '*.json', '*.hdf5', '*.pkl']:
            listOfFiles.extend(g.glob('binFiles/' + filetype))
        for filename in listOfFiles:
            temp_name = filename.split('/')[1]
            if not self.fs.exists(filename=temp_name):
                with io.FileIO(filename, 'r') as fileObject:
                    objectId = self.fs.put(fileObject, filename=temp_name)

    def get_file(self, filename):
        """
        Retrieve files from GridFS
        @param filename: String, filename of the file
        @return: GridOUT, filename returned
        """
        return self.fs.find({'filename': filename})

    def retr_init_file(self, filename):
        """
        Retrieve initial files from gridFS from the temp_fromfs folder
        @param filename: Name of the file
        @return: Object, the file
        """
        # Check if the temp folder exists
        if not os.path.exists(os.path.dirname('temp_fromfs/' + filename)):
            try:
                os.makedirs(os.path.dirname('temp_fromfs/' + filename))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        # Find the object and fetch it
        with open('temp_fromfs/' + filename, 'wb') as fileObject:
            try:
                fileObject.write(self.fs.find_one({'filename': filename}).read())
            except AttributeError:
                return None
        return filename

    def read_init_dataset(self, dataset):
        """
        Read the dataset from the local file fetched from gridfs
        @param dataset: String, Filename of the Dataset
        @return: Pandas DataFrame, the Dataset
        """
        return pd.read_hdf('temp_fromfs/' + dataset, 'rates')

    def load_local_files(self):
        """
        Load local files after training
        """
        listOfFiles = []
        for filetype in ['*.h5', '*.json', '*.hdf5', '*.pkl']:
            listOfFiles.extend(g.glob(filetype))
        for filename in listOfFiles:
            self.save_file_grid(filename)
            os.system('rm ' + filename)

    def save_file_grid(self, filename):
        """
        Save file in the gridfs from the local temp local system
        @param filename: String, the name of the file to be saved in the gridfs
        """
        if self.fs.find_one({'filename': filename}) is not None:
            self.fs.delete(self.fs.find_one({'filename': filename})._id)
        with io.FileIO(filename, 'r') as fileObject:
            objectId = self.fs.put(fileObject, filename=filename)
