import logging


class Logger:
    """
    A singleton logger class for OneSDK.
    This class ensures that only one logger instance is created and used throughout the application.
    """
    _instance = None

    def __new__(cls):
        """
        Create a new Logger instance if one doesn't exist, otherwise return the existing instance.
        This method implements the singleton pattern.

        Returns:
            Logger: The singleton Logger instance.
        """
        if cls._instance is None:
            # Create a new instance if one doesn't exist
            cls._instance = super(Logger, cls).__new__(cls)

            # Initialize the logger
            cls._instance._logger = logging.getLogger('OneSDK')
            cls._instance._logger.setLevel(logging.INFO)

            # Create a formatter for log messages
            cls._instance._formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Create and add a console handler
            ch = logging.StreamHandler()
            ch.setFormatter(cls._instance._formatter)
            cls._instance._logger.addHandler(ch)

        return cls._instance

    @classmethod
    def get_logger(cls):
        """
        Get the logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return cls()._logger

    @classmethod
    def set_debug_mode(cls, debug: bool = False):
        """
        Set the debug mode for the logger.

        Args:
            debug (bool): If True, set logger to DEBUG level. If False, set to INFO level.
        """
        logger = cls.get_logger()
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)


# Create a global logger instance
logger = Logger.get_logger()
