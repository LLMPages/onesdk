import logging
import os


class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._logger = logging.getLogger('OneSDK')
            cls._instance._logger.setLevel(logging.INFO)
            cls._instance._formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(cls._instance._formatter)
            cls._instance._logger.addHandler(ch)

        return cls._instance

    @classmethod
    def get_logger(cls):
        return cls()._logger

    @classmethod
    def set_debug_mode(cls, debug: bool = False):
        logger = cls.get_logger()
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)


logger = Logger.get_logger()