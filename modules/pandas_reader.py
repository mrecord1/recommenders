"""
Define a data processor class for use in reading text files into pandas and
then processing those dataframes for additional model information.
Including the dtypes is important for memory management and later processing.
"""

from typing import Text, Literal, Dict, Tuple
from collections import OrderedDict
import logging
import pandas as pd


class PandasReader:
    """ Accepts a schema and processes a text file into a pandas dataframe. """

    def __init__(
            self,
            schemas: Dict[Text, OrderedDict],
            schema_type: Literal["query", "candidate"] = "candidate"
            ):
        """
        Instantiate the PandasReader class.

        Args:
            schemas (dict): the schema for the text file.
            schema_type (literal): One of "query" or "candidate".
        """

        # sort the raw data because the merge is added out of order
        self.schema = schemas[schema_type]

        # get the names, dtypes, and id field for later use in pandas
        self.names = list(self.schema.keys())
        self.dtypes = [(key, values["dtype"])
                       for key, values in self.schema.items()]
        self.id_field = schema_type

        # set default values for id count and any vocabs
        self.df = None
        self.num = 0
        self.vocabs = {}

    def read_df(self, fileinfo: Dict[Text, dict]):
        """
        Read a text file and return information for a tensorflow model.

        Args:
            fileinfo (dict): the file information (filepath, header, sep).
        """

        # read in the data using the info from instantiation
        self.df = pd.read_csv(
            filepath_or_buffer=fileinfo["filepath"],
            header=fileinfo["header"],
            sep=fileinfo["sep"],
            names=self.names,
            dtype=self.dtypes
        )

    def find_num(self):
        """ Calc and log the number of unique ids. """

        self.num = self.df[self.id_field].nunique()
        logging.info("unique ids: %s: %s", self.id_field, self.num)

    def create_vocab(self):
        """ Calc the vocab fields as defined in the schema. """

        # create vocab fields from the schema
        skip_schema = ["drop", "q_emb_input", "c_emb_input", "c_numeric"]
        vocab_fields = [key for key, values in self.schema.items()
                        if values["tf_map"] not in skip_schema]

        # loop through the vocab fields
        for field in vocab_fields:

            # get the set of unique combinations of strings
            combos = set(self.df[field].values.ravel().astype('U').tolist())

            # split combos into unique keywords and add to vocabs
            keywords = {y for x in combos for y in x.split(" ") if y != ""}
            vocab = list(keywords)
            self.vocabs[field] = vocab

            # log the vocab
            logging.info("vocab %s: %s", field, vocab)

    def process(
            self,
            fileinfo: Dict[Text, dict]
            ) -> Tuple[pd.DataFrame, int, list]:
        """
        Orchestrate the steps in reading and profiling the data.

        Args:
            fileinfo (dict): the file information (filepath, header, sep).
        Returns:
            df, num, vocabs (tuple): the dataframe, number of ids, and vocabs.
        """

        self.read_df(fileinfo=fileinfo)
        self.find_num()
        self.create_vocab()

        return self.df, self.num, self.vocabs
