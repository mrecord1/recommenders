"""
Model definitions for each tower of a TwoTowersModel.
QueryModel and CandidateModel are submodels of TwoTowerModel.
Multiple Sequential models are used to allow for flexibility.
Projection layers are included later because they improve the model and ensure
the query and candidate dimensions match for the final dot product.
"""

from typing import Text, Dict
import tensorflow as tf
from keras import Sequential
from keras.layers import (TextVectorization, StringLookup, Embedding,
                          GlobalAveragePooling1D, Flatten)
from keras.models import Model


class QueryModel(Model):
    """ Flexible Query submodel for the two tower model. """

    def __init__(self, data: dict, hp: dict):
        """
        Instantiate the query model with required info.

        Args:
            data (dict): dictionary of model building data (e.g., vocab).
            hp (dict): dictionary of hyperparameters.
        """

        super().__init__()

        self.data = data
        self.hp = hp

        if hp["use_q_color"]:
            self.q_color = Sequential([
                # all colors are known, so no oov
                StringLookup(
                    vocabulary=data["vocabs_q"]["color_family"],
                    num_oov_indices=0,
                    name="q_color_lookup"
                ),
                Embedding(
                    input_dim=hp["n_colors"],
                    output_dim=hp["dims_q_color"],
                    embeddings_regularizer="l2",
                    name="q_color_emb"
                ),
                Flatten(name="q_color_flatten")
            ])

        # vocab should come from candidate data for full coverage
        if hp["use_q_oracle"]:
            self.q_oracle = Sequential([
                TextVectorization(
                    split="whitespace",
                    output_mode="int",
                    output_sequence_length=hp["max_len_oracle"],
                    vocabulary=data["vocabs_c"]["oracle_categories_x"],
                    name="q_oracle_vec"
                ),
                # include 2 extra dimensions for padding and oov
                Embedding(
                    input_dim=len(data["vocabs_c"]["oracle_categories_x"]) + 2,
                    output_dim=hp["dims_q_oracle"],
                    embeddings_regularizer="l2",
                    input_length=hp["max_len_oracle"],
                    name="q_oracle_emb"
                ),
                GlobalAveragePooling1D(name="q_oracle_pooling")
            ])

        if hp["use_q_latent"]:
            self.q_emb = Sequential([
                # include 1 extra dimensions for oov
                Embedding(
                    input_dim=hp["n_q"] + 1,
                    output_dim=hp["dims_q_latent"],
                    embeddings_regularizer="l2",
                    name="q_latent_emb"
                ),
                Flatten(name="q_latent_flat")
            ])

    def call(
            self,
            inputs: Dict[Text, tf.Tensor],
            training=None,
            mask=None
            ) -> tf.Tensor:
        """
        Call query model, map inputs to submodels and concat results.

        Args:
            inputs (dict): dictionary of model inputs.
            training (bool): whether to train or not.
            mask (bool): mask or list of masks.
        Returns:
            tf.Tensor: the model output.
        """

        concats = []
        if self.hp["use_q_color"]:
            concats.append(self.q_color(inputs["q_color"]))

        if self.hp["use_q_oracle"]:
            concats.append(self.q_oracle(inputs["q_oracle"]))

        if self.hp["use_q_latent"]:
            concats.append(
                self.q_emb(inputs["q_emb_input"])
            )

        return tf.concat(concats, axis=1)


class CandidateModel(Model):
    """ Candidate model for two tower model. """

    def __init__(self, data: dict, hp: dict):
        """
        Instantiate the candidate model with required data and hyperparameters.

        Args:
            data (dict): dictionary of model building data (e.g., vocab).
            hp (dict): dictionary of hyperparameters.
        """

        super().__init__()

        self.data = data
        self.hp = hp

        if hp["use_c_latent"]:
            self.c_emb = Sequential([
                # include 1 extra dimensions for oov
                Embedding(
                    input_dim=hp["n_c"] + 1,
                    output_dim=hp["dims_c_latent"],
                    embeddings_regularizer="l2",
                    name="c_emb"
                ),
                Flatten(name="c_latent_flat")
            ])

        if hp["use_c_subtype"]:
            self.c_subtype = Sequential([
                TextVectorization(
                    split="whitespace",
                    output_mode="int",
                    output_sequence_length=hp["max_len_subtype"],
                    vocabulary=data["vocabs_c"]["subtype"],
                    name="c_subtype_vec"
                ),
                # include 2 extra dimensions for padding and oov
                Embedding(
                    input_dim=len(data["vocabs_c"]["subtype"]) + 2,
                    output_dim=hp["dims_c_subtype"],
                    embeddings_regularizer="l2",
                    input_length=hp["max_len_subtype"],
                    name="c_subtype_emb"
                ),
                GlobalAveragePooling1D(name="c_subtype_pooling")
            ])

        if hp["use_c_oracle"]:
            self.c_oracle = Sequential([
                TextVectorization(
                    split="whitespace",
                    output_mode="int",
                    output_sequence_length=hp["max_len_oracle"],
                    vocabulary=data["vocabs_c"]["oracle_categories_x"],
                    name="c_oracle_vec"
                ),
                # include 2 extra dimensions for padding and oov
                Embedding(
                    input_dim=len(data["vocabs_c"]["oracle_categories_x"]) + 2,
                    output_dim=hp["dims_c_oracle"],
                    embeddings_regularizer="l2",
                    input_length=hp["max_len_oracle"],
                    name="c_oracle_emb"
                ),
                GlobalAveragePooling1D(name="c_oracle_pooling")
            ])

    def call(
            self,
            inputs: Dict[Text, tf.Tensor],
            training=None,
            mask=None
            ) -> tf.Tensor:
        """
        Call candidate model on inputs.

        Args:
            inputs (dict): dictionary of model inputs.
            training (bool): whether to train or not.
            mask (bool): mask or list of masks.
        Returns:
            tf.Tensor: the model output.
        """

        concats = []
        if self.hp["use_c_latent"]:
            concats.append(
                self.c_emb(inputs["c_emb_input"])
            )

        if self.hp["use_c_numeric"]:
            concats.append(tf.cast(inputs["c_numeric"], tf.float32))

        if self.hp["use_c_subtype"]:
            concats.append(self.c_subtype(inputs["c_subtype"]))

        if self.hp["use_c_oracle"]:
            concats.append(self.c_oracle(inputs["c_oracle"]))

        return tf.concat(concats, axis=1)
