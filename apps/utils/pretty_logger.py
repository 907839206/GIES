
import logging
import pprint

from .singleton import singleton

@singleton
class PrettyLogger:
    def __init__(self):
        self.__logger = self.__create_logger()

    def __create_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            logger.removeHandler(handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger

    def __pretty_log(self, log):
        return pprint.pformat(log, indent=4)

    def info(self, info):
        pretty_info = self.__pretty_log(info)
        self.__logger.info(pretty_info)

    def debug(self, debug):
        self.__logger.debug(self.__pretty_log(debug))

    def setLevel(self, level):
        self.__logger.setLevel(level)

    def error(self,error):
        self.__logger.error(self.__pretty_log(error))

logger = PrettyLogger()