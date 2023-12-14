"""
Model definition of the Tensorflow Two Tower Model with both ranking and
retrieval tasks. Projection layers are included because they improve the model
and ensure the query and candidate dimensions match for the final dot product.
"""

from typing import Text, Dict
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow_recommenders.layers import factorized_top_k
from tensorflow_recommenders.metrics import FactorizedTopK
from keras import Sequential
from keras.layers import Dense, Layer
from keras.models import Model

from modules.towers import QueryModel, CandidateModel


class TwoTowerModel(tfrs.Model):
    """ Two tower recommendation model. """

    def __init__(self, data: dict, hp: dict):
        """
        Instantiate the two tower model with required info.

        Args:
            data (dict): dictionary of model building data (e.g., vocab).
            hp (dict): dictionary of hyperparameters.
        """

        super().__init__()

        # data and hyperparameters
        self.data = data
        self.hp = hp

        # add the query model with tunable projection layers and activation
        self.query_model: Model = Sequential(
            [QueryModel(data, hp)]
            + [Dense(
                hp[f"dims_proj_{layer}"],
                activation=hp["activation_proj"]
                ) for layer in range(hp["n_proj_layers"])]
            )

        # add the candidate model with tunable projection layers and activation
        self.candidate_model: Model = Sequential(
            [CandidateModel(data, hp)]
            + [Dense(
                    hp[f"dims_proj_{layer}"],
                    activation=hp["activation_proj"]
               ) for layer in range(hp["n_proj_layers"])]
        )

        # include a ranking model with tunable ranking layers and activation
        self.ranking_model: Model = Sequential(
           [Dense(
                 hp[f"dims_rank_{layer}"],
                 activation=hp["activation_rank"]
            ) for layer in range(hp["n_rank_layers"])]
           + [Dense(1)]
        )

        # metrics, tasks, and default index of none
        self.t_metrics = FactorizedTopK(
            candidates=data["index_data"]
            .batch(hp["batch_size_index"])
            .map(self.candidate_model)
        )

        self.ranking_task: Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

        self.retrieval_task: Layer = tfrs.tasks.Retrieval(
            metrics=self.t_metrics
        )

        self.index = None

    def call(
            self,
            inputs: Dict[Text, tf.Tensor],
            training=None,
            mask=None
            ) -> tuple:
        """
        Call the two tower model.

        Args:
            inputs (dict): dictionary of model inputs.
            training (bool): whether to train or not.
            mask (bool): mask or list of masks.
        Returns:
            outputs (tuple): the embeddings and ranking predictions.
        """

        # query embeddings
        q_inputs = {k: v for k, v in inputs.items() if k[0] == "q"}
        q_embs = self.query_model(q_inputs)

        # positive candidate embeddings
        c_inputs = {k: v for k, v in inputs.items() if k[0] == "c"}
        c_embs = self.candidate_model(c_inputs)

        # calculate the embeddings dot product as a ranking feature
        dot_product = tf.reduce_sum(q_embs * c_embs, axis=1, keepdims=True)
        ranking_features = tf.concat([q_embs, c_embs, dot_product], axis=1)

        return q_embs, c_embs, self.ranking_model(ranking_features)

    def compute_loss(
            self,
            inputs: Dict[Text, tf.Tensor],
            training=False
            ) -> tf.Tensor:
        """
        Compute loss for two tower model.

        Args:
            inputs (dict): dictionary of model inputs.
            training (bool): whether to train or not.
            mask (bool): mask or list of masks.
        Returns:
            loss: (tf.Tensor): the total loss.
        """

        # get rankings
        rankings = inputs.pop("ranking")

        # get embeddings and predicted rankings
        query_embs, candidate_embs, ranking_preds = self(inputs)

        # calc ranking loss
        ranking_loss = self.ranking_task(
            labels=rankings,
            predictions=ranking_preds,
        )

        # calc retrieval loss
        retrieval_loss = self.retrieval_task(query_embs, candidate_embs)

        return (
            ranking_loss * (1 - self.hp["loss_weight_retrieval"]) +
            retrieval_loss * self.hp["loss_weight_retrieval"]
        )

    def build_index(
            self,
            inputs: Dict[Text, tf.Tensor]
            ) -> factorized_top_k.BruteForce:
        """
        Predict method for two tower model.

        Args:
            inputs (dict): dictionary of model inputs.
        Returns:
            retrieval index (BruteFroce): the index.
        """

        # define brute force index
        self.index = factorized_top_k.BruteForce(
            query_model=self.query_model,
            k=self.hp["k"]
        )

        # transform data for brute force indexing
        identifiers = inputs \
            .map(lambda x: x["c_emb_input"]) \
            .batch(self.hp["batch_size_index"])

        embs = inputs \
            .batch(self.hp["batch_size_index"]) \
            .map(self.candidate_model)

        zipped = tf.data.Dataset.zip((identifiers, embs))
        self.index.index_from_dataset(zipped)
