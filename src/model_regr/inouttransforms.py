
import functools, time, yaml, json
from flask import Response
from model_regr.logger import PlainnLogger
import pandas as pd

LOG = PlainnLogger.getLogger(__name__)


def catch_exception(f):
    @functools.wraps(f)
    def func(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except ValueError as e:
            print('Caught an exception in',f.__name__)

    return func

class InOutTransforms:

    def dict_generator(self, indict,pre=None):
        pre = pre[ : ] if pre else [ ]
        if isinstance(indict,dict):
            for key, value in indict.items():
                if isinstance(value,dict):
                    for d in self.dict_generator(value, [key]+pre):
                        yield d
                elif isinstance(value,list) or isinstance(value, tuple):
                    for v in value:
                        for d in self.dict_generator(v, [key]+pre):
                            yield d
                else:
                    yield pre+[key, value]
        else:
            yield indict

    @catch_exception
    def yaml_d(self, content):
        try:
            return yaml.duml(content)
        except ValueError as e:
            return self.generic_resp(400, "application/x-yaml", "InvalidArgument:" + str(e))

    @catch_exception
    def json_d(self, content):
        try:
            return json.dumps(content)
        except ValueError as e:
            return self.generic_resp(400, "application/json", "InvalidArgument:" + str(e))

    @catch_exception
    def generic_err400_with_resp(self,message,start_time):
        LOG.error(message, extra={"status": 400, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})
        return self.generic_resp(400, 'application/json',
                                  self.json_d(self.error_message(message)))
    @catch_exception
    def yaml_l(self, content):
        try:
            return yaml.load(content)
        except ValueError as e:
            return self.generic_resp(400, "application/x-yaml", "InvalidArgument:" + str(e))

    def conv_json_to_df(self,json_dump):
        if type(json_dump) == dict:
            return pd.DataFrame(pd.Series(json_dump)).transpose()
        else:
            return pd.DataFrame.from_records(json_dump)


    @catch_exception
    def error_message(self, message):
         return {'Error': message}

    @catch_exception
    def success_message(self, message):
         return {'Success': message}

    @catch_exception
    def json_l(self, content):
        try:
            return json.loads(content)
        except ValueError as e:
            return self.generic_resp(400, "application/json", "InvalidArgument:" + str(e))

    def generic_resp(self, status, content_type, response):
        return Response(status = status,
                        content_type = content_type,
                        response = response)

    def transform_text(self, request):
        if request.content_type == "application/json":
            return self.json_l(request.data)
        elif request.content_type == "application/x-yaml":
            return self.yaml_l(request.data)
        else:
            return False
