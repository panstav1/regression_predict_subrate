# Libraries
import functools, time, json
from flask import Response
from model_regr.logger import PlainnLogger
import pandas as pd

# Logger
LOG = PlainnLogger.getLogger(__name__)


def catch_exception(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            print('Caught an exception in', f.__name__)

    return func


class InOutTransforms:
    """
    InOutTransforms Class
    Class instance of defining Input and Output Transform of the overall messages
    """

    @catch_exception
    def json_d(self, content):
        """
        Dump model with try except clause
        @param content: dump content of json
        @return: string returned
        """
        try:
            return json.dumps(content)
        except ValueError as e:
            return self.generic_resp(400, "application/json", "InvalidArgument:" + str(e))

    @catch_exception
    def generic_err400_with_resp(self, message, start_time):
        """
        Generic response with Log and response to the client
        @param message: Meesage to be displayed in the Log function and the output of the endpoint
        @param start_time: counter of start time in the end point
        @return: Return of the Logging and the response with 400 Bad Content
        """
        LOG.error(message, extra={"status": 400, "time_elapsed": "%.3f seconds" % (time.time() - start_time)})
        return self.generic_resp(400, 'application/json',
                                 self.json_d(self.error_message(message)))

    @catch_exception
    def conv_json_to_df(self, json_dump):
        """
        Convert JSON to dataframe with one or multiple records
        @param json_dump: String from the json dump
        @return: Return dataframe either of one single or multiple records
        """
        if type(json_dump) == dict:
            return pd.DataFrame(pd.Series(json_dump)).transpose()
        else:
            return pd.DataFrame.from_records(json_dump)


    @catch_exception
    def error_message(self, message):
        """
        Return error message JSON with Error key for response
        @param message: String message
        @return: Return Dict with Error key in front
        """
        return {'Error': message}

    @catch_exception
    def success_message(self, message):
        """
        Return error message JSON with Success key for response
        @param message: String message
        @return: Return Dict with Success key in front
        """
        return {'Success': message}

    @catch_exception
    def json_l(self, content):
        """
        JSON loads with try except clause
        @param content: Content of json string format
        @return: JSON object loaded from the string-like JSON format
        """
        try:
            return json.loads(content)
        except ValueError as e:
            return self.generic_resp(400, "application/json", "InvalidArgument:" + str(e))

    @catch_exception
    def generic_resp(self, status, content_type, response):
        """
        Generic response to the client
        @param status: Code of the response
        @param content_type: Content-type of the response
        @param response: Plain response text
        @return:
        """
        return Response(status=status,
                        content_type=content_type,
                        response=response)
