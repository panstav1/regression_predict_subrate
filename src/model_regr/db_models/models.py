
import pymongo, h5py
import io, glob as g, pandas as pd, sys
import pandas as pd
from datetime import datetime
import pymongo, os, gridfs, uuid, functools
from bson.errors import InvalidId
from bson import ObjectId
import os
import errno




def catch_Type_exception(f):
    @functools.wraps(f)
    def func(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except TypeError as e:
            print('Caught an exception in',f.__name__)
        return func


class MongoDB:
    """
    DB class
    Class instance is a collection (specified in __init__) to perform operations on
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

    def __init__(self, collection,):
        self.host = os.environ.get("MONGO_URL", self.def_host)
        self.port = os.environ.get("MONGO_PORT", self.def_port)
        self.database = self.def_database
        self.conn = pymongo.MongoClient(self.MONGO_URI.format(self.host, self.port))
        self.collection = self.conn[self.database][collection]
        self.fs = gridfs.GridFS(self.conn[self.database])
        self.db = self.conn[self.database]
        self.load_init_files()



        # f = pd.read_hdf('rates.h5', 'rates', "r")
        # xx = json.loads(pd.concat([f.drop(f.columns[3:-1],axis=1),
        #                            pd.Series(f.iloc[:, 3:-1].to_numpy().tolist(),name='features')],
        #                           axis=1).rename({'surveyID':'_id'},axis=1)
        #                 .to_json(orient="records"))

        # start=time.time()
        # aa = self.collection.insert_many(xx)
        # print(time.time() - start)
        # result = f.iloc[1:3,:].to_json(orient="records")
        # parsed = json.loads(result)
        # print(parsed)

    def load_init_files(self):
        listOfFiles = []
        for filetype in ['*.h5', '*.json', '*.hdf5', '*.pkl']:
            listOfFiles.extend(g.glob('binFiles/' + filetype))
        for filename in listOfFiles:
            temp_name = filename.split('/')[1]
            if not self.fs.exists(filename=temp_name):
                with io.FileIO(filename, 'r') as fileObject:
                    objectId = self.fs.put(fileObject, filename=temp_name)


    def get_file(self, filename):
        return self.fs.find({'filename': filename})


    def retr_init_file(self, filename):
        if not os.path.exists(os.path.dirname('temp/' + filename)):
            try:
                os.makedirs(os.path.dirname('temp/' + filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open('temp/' + filename, 'wb') as fileObject:
            fileObject.write(
                             self.fs.find_one(
                                 {'filename': filename}).read())
        return filename

    def insert_grid(self, model):
        with open(model,'rb') as dictionary:
            fileID = self.fs.put(dictionary)
        os.remove(model)
        return fileID




    def insert_grid(self, model):
        with open(model,'rb') as dictionary:
            fileID = self.fs.put(dictionary)
        os.remove(model)
        return fileID

    def load_grid(self, model_id, filename):
        fileID = self.fs.get(model_id)
        f = open(str(filename),"wb")
        f.write(fileID.read())

    def find_delete_model_metadata(self, model):
        return self.collection.find_one_and_delete({'filename': model})

    def find_user_bool(self, user):
        return self.collection.find({"user":  user}).count() == 1;

    def find_delete_model(self, model_id):
        return self.fs.delete(model_id)

    def get_model_metadata(self, model):
        return self.collection.find_one({'filename': model})

    def check_exist_model(self, model):
        return self.collection.find({'filename': model}).count() == 1

    def gen_uuid(self):
        return str(uuid.uuid1())

    def insert_user(self, user, test):
        self.collection.update({self.ID: self.dict_users},
                                       {'$push': {'user': user ,'item': test ,'rating': 1 }}, upsert=True)
        self.close()

    def delete_user(self, delete_item, array):
        dic_user = self.collection.find_one({self.ID: self.dict_users})
        duplicates = dic_user[array].count(delete_item)
        for _ in range(duplicates):
            index = dic_user[array].index(delete_item)
            del dic_user['user'][index]
            del dic_user['item'][index]
            del dic_user['rating'][index]
        self.close()
        return dic_user, duplicates

    def delete_dictuser(self):
        return self.collection.delete_one({self.ID: self.dict_users})

    def get_dict(self):
        return self.collection.find_one({self.ID: self.dict_users})

    def insert_dictuser(self, dict):
        return self.collection.insert_one(dict)

    def find_users(self):
        """Returns document if _id in collection else None"""
        try:
            dic_user = self.collection.find_one({self.ID: self.dict_users})
            del dic_user[ self.ID ]
            return dic_user
        except InvalidId:
            return

    def insert_model_metadata(self, uuid, filename):
        self.collection.insert_one(
            {self.ID: uuid,
             'filename': filename})

    def insert_one_file(self):
        self.collection.insert_one(
            {self.ID: self.gen_uuid(),
             'asda':'asdasd'})

    def add_user_to_collection(self, email):
        """
        Check if param `email` in collection.
        If true, return its document's token, else inserts it to collection and returns its document's
        token.
        :param email: Email address to look for in collection.
        """
        token = uuid.uuid4().hex
        if self.find_by_email(email):  # Validation that user is not None
            return self.randomize_token(email)
        else:
            self.collection.insert_one({
                self.EMAIL: email.strip(),
                self.GENERATED_COUNT: 1,
                self.VERIFIED: False,
                self.LAST_USED: datetime.now(),
                self.TOKEN: token
            })
        return token

    def find_by_id(self, uid):
        """Returns document if _id in collection else None"""
        try:
            return self.collection.find_one({self.ID: uid})
        except InvalidId:
            return

    def find_by_token(self, token):
        return self.collection.find_one({self.TOKEN: token})

    def is_verified(self, uid):
        """Check if user's verified status in set to True"""
        try:
            verified = self.find_by_id(uid).get(self.VERIFIED)
            return verified
        except AttributeError:
            return False

    def randomize_token(self, email):
        """Randomize the token after user validation"""
        token = uuid.uuid4().hex
        self.collection.find_one_and_update(
            {self.EMAIL: email}, {self.SET: {self.TOKEN: token}}
        )
        return token

    def update_user(self, token, verification=False):
        """
        Changes user verification status to `True` if verification boolean is set.
        Otherwise, increments generated_count and updates last_use to datetime.now().
        """
        if verification:
            self.collection.find_one_and_update(
                {self.TOKEN: token}, {self.SET: {self.VERIFIED: True}}
            )
        else:
            self.collection.find_one_and_update(
                {self.TOKEN: token}, {self.SET: {self.LAST_USED: datetime.now()},
                                      self.INC: {self.GENERATED_COUNT: 1}}
            )
        return

    def close(self):
        self.conn.close()
