import json
from typing import Any, Dict, Optional

class Config:
    """
    A configuration management class that provides a flexible interface for storing and retrieving configuration values.

    This class supports dictionary-style access, file I/O operations, and various utility methods for managing configuration data.

    Attributes:
        _config (Dict[str, Any]): The internal dictionary storing configuration key-value pairs.

    Example:
        >>> config = Config({"debug": True, "api_key": "abc123"})
        >>> config.get("debug")
        True
        >>> config["api_key"] = "new_key"
        >>> config.as_dict
        {'debug': True, 'api_key': 'new_key'}
    """

    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new Config instance.

        Args:
            initial_config (Optional[Dict[str, Any]]): Initial configuration dictionary. Defaults to None.
        """
        self._config: Dict[str, Any] = initial_config or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value.

        Args:
            key (str): The configuration key to retrieve.
            default (Any): The default value to return if the key is not found. Defaults to None.

        Returns:
            Any: The configuration value associated with the key, or the default value if not found.
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key (str): The configuration key to set.
            value (Any): The value to associate with the key.
        """
        self._config[key] = value

    def update(self, new_config: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.

        Args:
            new_config (Dict[str, Any]): A dictionary of configuration key-value pairs to update.
        """
        self._config.update(new_config)

    def remove(self, key: str) -> None:
        """
        Remove a configuration item.

        Args:
            key (str): The key of the configuration item to remove.
        """
        self._config.pop(key, None)

    def clear(self) -> None:
        """Clear all configuration items."""
        self._config.clear()

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to configuration values.

        Args:
            key (str): The configuration key to retrieve.

        Returns:
            Any: The configuration value associated with the key.

        Raises:
            KeyError: If the key is not found in the configuration.
        """
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-style setting of configuration values.

        Args:
            key (str): The configuration key to set.
            value (Any): The value to associate with the key.
        """
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration key exists.

        Args:
            key (str): The configuration key to check.

        Returns:
            bool: True if the key exists in the configuration, False otherwise.
        """
        return key in self._config

    def __repr__(self) -> str:
        """
        Return a string representation of the Config instance.

        Returns:
            str: A string representation of the Config instance.
        """
        return f"Config({self._config})"

    @property
    def as_dict(self) -> Dict[str, Any]:
        """
        Return a copy of the configuration as a dictionary.

        Returns:
            Dict[str, Any]: A copy of the internal configuration dictionary.
        """
        return self._config.copy()

    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from a JSON file.

        Args:
            file_path (str): The path to the JSON file to load.

        Raises:
            json.JSONDecodeError: If the file contains invalid JSON.
            IOError: If there's an error reading the file.
        """
        with open(file_path, 'r') as f:
            self._config.update(json.load(f))

    def save_to_file(self, file_path: str) -> None:
        """
        Save the current configuration to a JSON file.

        Args:
            file_path (str): The path to the file where the configuration will be saved.

        Raises:
            IOError: If there's an error writing to the file.
        """
        with open(file_path, 'w') as f:
            json.dump(self._config, f, indent=2)


# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    'debug': False,
    'timeout': 30,
    'max_retries': 3,
    'retry_delay': 1,
}

# Global configuration instance
global_config: Config = Config(DEFAULT_CONFIG)