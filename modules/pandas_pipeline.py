"""
Defines a pipeline for processing a pandas dataframe-based input.
Pipeline steps are flexible and can be used independently.
Steps include feature engineering, preprocessing (e.g., scaling),
conversion, and orchestration.
"""

from typing import Tuple, Literal
from collections import defaultdict, OrderedDict
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PandasPipeline:
    """ Perform a pipeline of operations on a pandas dataframe. """

    def __init__(self, schema: OrderedDict, hp: dict):
        """
        Instantiate a pipeline for pandas data.

        Args:
            schema (OrderedDict): the schema for the dataframe.
            hp (dict): hyperparameters for the pipeline.
        """

        self.schema = schema
        self.hp = hp["fixed"]

    def calc_rankings(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer a feature for frequency (implied rankings).

        Args:
            data (pd.DataFrame): dataframe to calculate frequencies for.
        Returns:
            data (pd.DataFrame): the updated dataframe.
        """

        # process denominator
        data["key"] = (data["user"].astype(str) + ";"
                       + data["query"].astype(str))
        train_denom = data["key"].value_counts()
        data["denom"] = data["key"].map(train_denom)

        # process numerator
        data["pair"] = (data["query"].astype(str) + ";"
                        + data["name"])
        train_num = data["pair"].value_counts()
        data["num"] = data["pair"].map(train_num)

        # get frequecies as quantiles (1 to 5 scale)
        data["freq"] = data["num"] / data["denom"]
        labels = range(1, 6)
        data["ranking"] = pd.qcut(data['freq'], q=5, labels=labels)

        # drop unnecessary columns and update schema
        data = data.drop(columns=["key", "pair", "num", "denom", "freq"])
        self.schema["ranking"] = {
            "seq": 28, "dtype": "int64", "tf_map": "ranking"
        }

        return data

    def preprocess_data(
            self,
            training_data: pd.DataFrame,
            test_data: pd.DataFrame
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the training data (scale numeric columns).

        Args:
            training_data (pd.DataFrame): the training dataframe to process.
            test_data (pd.DataFrame): the test dataframe to process.
        Returns:
            training_data, test_data (tuple): the processed data.
        """

        # get numeric columns
        numeric_fields = [key for key, values in self.schema.items()
                          if values["tf_map"] == "c_numeric"]

        # instantiate the scaler
        scaler = StandardScaler()

        # fit and transform numeric columns in the training data
        training_data[numeric_fields] = scaler.fit_transform(
            training_data[numeric_fields]
        )

        # transform numeric columns in the test data
        test_data[numeric_fields] = scaler.transform(
            test_data[numeric_fields]
        )

        return training_data, test_data

    def convert_to_tf(
            self,
            data: pd.DataFrame,
            dataset_type: Literal["train", "test", "index"]
            ) -> tf.data.Dataset:
        """
        Converts a dataframe to a tensorflow dataset.

        Args:
            data (pd.DataFrame): the dataframe to convert.
            dataset_type (literal): one of 'train', 'test', or 'index'.
        Returns:
            tf_dataset (tf.data.Dataset): the data as a tensorflow dataset.
        """

        # convert the pd schema to map each field to a tf dataset schema
        tf_groups = defaultdict(list)
        for key, values in self.schema.items():
            tf_groups[values["tf_map"]].append(key)

        # create the tensorflow schema, and include the relevant data
        tf_schema = {group: data[fields].values
                     for group, fields in tf_groups.items() if group != "drop"}

        # create the tensorflow dataset from the schema and data
        tf_dataset = tf.data.Dataset.from_tensor_slices(tf_schema)

        # get a count for shuffling
        n = sum(1 for _ in tf_dataset)

        # re-shuffle train
        if dataset_type == "train":
            tf_dataset = tf_dataset.shuffle(n, seed=self.hp["seed"])

        # batch and cache train and test
        if dataset_type != "index":
            tf_dataset = tf_dataset \
                .batch(self.hp[f"batch_size_{dataset_type}"]) \
                .cache()

        return tf_dataset

    def run_pipeline(
            self,
            data: pd.DataFrame
            ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Run the pipeline.

        Args:
            data (pd.DataFrame): the dataframe to process.
        Returns:
            cached_train, cached_test (tuple): the cached tf data.
        """

        # train test split
        train, test = train_test_split(
            data, test_size=self.hp["test_ratio"], random_state=self.hp["seed"]
        )

        # add rankings to train and test data
        train = self.calc_rankings(train)
        test = self.calc_rankings(test)

        # preprocess
        train, test = self.preprocess_data(train, test)

        # convert to tf
        cached_train = self.convert_to_tf(train, dataset_type="train")
        cached_test = self.convert_to_tf(test, dataset_type="test")

        return cached_train, cached_test
