"""
Yaml readers for reading yaml files.
Primary focus is on reading config and schema files.
"""

from typing import Dict, Tuple, TextIO
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
import yaml


class YamlReader(ABC):
    """ Reads a yaml file. """

    def __init__(self, filepath: TextIO):
        """
        Reads a yaml file.

        Args:
            filepath (TextIO): The filepath of the config file.
        """
        try:
            with open(filepath, "r", encoding="utf8") as f:
                self.data = yaml.safe_load(f)

            logging.info('file read: %s', filepath)

        except FileNotFoundError as e:
            raise e

        except yaml.YAMLError as e:
            raise e

    @abstractmethod
    def validate_file(self):
        """ Validate the file. """
        pass

    @abstractmethod
    def process_file(self):
        """ Process the file. """
        pass


class ConfigYamlReader(YamlReader):
    """ Read a yaml file for configuration details. """

    def __init__(self, filepath: TextIO):
        """
        Reads config details. Assumes yaml first level is "params", "paths".

        Args:
            filepath (TextIO): The filepath of the config file.
        """

        super().__init__(filepath=filepath)

    def validate_file(self):
        """ Validate the config file. """

        if tuple(self.data.keys()) != ("params", "paths"):
            e = "config file must have 'params' and 'paths'"
            logging.error(e)
            raise ValueError(e)

    def process_file(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Process the config file to get hyperparameters and paths.
        Assumes yaml first level is "params", "paths".

        Returns:
            output (tuple): hyperparameters (dict) and paths (dict).
        """

        # validate the file
        self.validate_file()

        # get hyperparameters and paths
        hyperparameters = self.data.get("params", {})
        config_paths = self.data.get("paths", {})

        # log hyperparameters and paths
        logging.info("hyperparameters: %s", hyperparameters)
        logging.info("config_paths: %s", config_paths)

        return hyperparameters, config_paths


class SchemaYamlReader(YamlReader):
    """ Read a yaml file for schema details. """

    def __init__(self, filepath: TextIO):
        """
        Reads a yaml file for config details.

        Args:
            filepath (TextIO): The filepath of the schema file.
            Assumes field schema includes "seq", "dtype", "tf_map".
        """

        super().__init__(filepath=filepath)

    def validate_file(self):
        """ Validate the schema file. """

        # validate that the first level is "candidate" and "query"
        if tuple(self.data.keys()) != ("candidate", "query"):
            e = "config file must have 'candidate' and 'query'"
            logging.error(e)
            raise ValueError(e)

        # validate that each field has "seq", "dtype", "tf_map"
        else:
            required = ["seq", "dtype", "tf_map"]
            for key in self.data.keys():
                for field in self.data[key].keys():
                    search = list(self.data[key][field].keys())
                    if not all(item in search for item in required):
                        e = "field schema must have 'seq', 'dtype', 'tf_map'"
                        logging.error(e)
                        raise ValueError(e)

    def process_file(self) -> Dict[str, OrderedDict]:
        """
        Process the schema file. Returns a dictionary of ordered dictionaries.
        Assumes field schema includes "seq", "dtype", "tf_map".

        Returns:
            schemas (dict): a dictionary of ordered dictionaries.
        """

        # validate the file
        self.validate_file()

        # get the schemas
        schemas = {}
        for key in self.data.keys():
            items = self.data[key].items()
            schema = OrderedDict(sorted(items, key=lambda i: i[1]["seq"]))
            schemas[key] = schema

            # log
            logging.info("schema: %s", schema)

        return schemas
