"""
Defines a hypermodel for use in keras tuner and a ModelTuner class for
launching hyperparameter tuning. The hyperparametes are defined in a yaml
file and converted to keras tuner format by type.
"""

import logging
from keras_tuner import HyperModel, HyperParameters, BayesianOptimization
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adagrad

from modules.models import TwoTowerModel


class TFRSHyperModel(HyperModel):
    """ Build the two tower hypermodel for hyperparameter tuning. """

    def __init__(self, data: dict):
        """
        Instantiate the TFRS hypermodel with required data.

        Args:
            data (dict): dictionary of model building data (e.g., vocab).
        """

        super().__init__()

        self.data = data

    def build(self, hp: HyperParameters) -> TwoTowerModel:
        """
        Build and compile the two tower model.

        Args:
            hp (HyperParameters): keras hyperparameters for the model.
        Returns:
            TwoTowerModel (TwoTowerModel): the compiled two tower model.
        """

        # instantiate the model
        model = TwoTowerModel(data=self.data, hp=hp)

        # compile the model with Adagrad optimizer
        model.compile(optimizer=Adagrad(learning_rate=hp["lr"]))

        return model


class ModelTuner:
    """ Find the best model by tuning hyperparameters. """

    def __init__(self, data: dict, hp: dict, paths: dict):
        """
        Instantiate the model tuner with required info.

        Args:
            data (dict): dictionary of model building data (e.g., vocab).
            hp (dict): dictionary of hyperparameters.
            paths (dict): dictionary of file paths.
        """

        self.model = TFRSHyperModel(data=data)
        self.data = data
        self.hp = hp
        self.paths = paths
        self.khp = None
        self.best_model = None

    def convert_conditional(self, key: str, values: dict):
        """
        Convert conditional hp to keras tuner format.
        
        Args:
            key (str): the hyperparameter key (name).
            values (dict): the hyperparameter values.
        """

        # use a conditional scope (if statement still required)
        with self.khp.conditional_scope(values["cond"], True):

            # if the condition is true, add to keras hp
            if values["cond"]:

                # numeric is the only fixed value
                if key == "dims_c_numeric":
                    self.khp.Fixed(key, values["value"])
                else:
                    self.khp.Int(
                        key,
                        min_value=values["min"],
                        max_value=values["max"],
                        step=values["step"]
                    )

    def convert_hyperparameters(self):
        """ Convert hyperparameters to keras tuner format. """

        # define storage of hyperparameters in keras tuner format
        self.khp = HyperParameters()

        # loop through the nested list of hyperparameters, matching on type
        for hp_type, hps in list(self.hp.items()):
            match hp_type:

                # for each bool, add key and value to keras hp
                case "bool":
                    for key in hps:
                        self.khp.Boolean(key, default=hps[key])

                # for each int, add key and values to keras hp
                case "int":
                    for key in hps:
                        values = hps[key]
                        self.khp.Int(
                            key,
                            min_value=values["min"],
                            max_value=values["max"],
                            step=values["step"]
                        )

                # for each float, add key and values to keras hp
                case "float":
                    for key in hps:
                        values = hps[key]
                        self.khp.Float(
                            key,
                            min_value=values["min"],
                            max_value=values["max"],
                            sampling=values["sampling"]
                        )

                # for each fixed, add key and values to keras hp
                case "fixed":
                    for key in hps:
                        self.khp.Fixed(key, value=hps[key])

                # for each choice, add key and values to keras hp
                case "choice":
                    for key in hps:
                        self.khp.Choice(key, values=hps[key], ordered=False)

                # for each conditional, evaluate and add if appropriate
                case "conditional":
                    for key in hps:
                        values = hps[key]
                        self.convert_conditional(key, values)

    def tune_model(self):
        """ Run the hyperparameter tuner with Baresian Optimization. """

        # define the tuner with the previously defined hyperparameters
        tuner = BayesianOptimization(
            hypermodel=self.model,
            objective="loss",
            max_trials=self.khp["max_trials"],
            seed=self.khp["seed"],
            hyperparameters=self.khp,
            directory="logs",
            project_name="tfrs",
            overwrite=True,
        )

        # run the tuner
        tuner.search(
            self.data["cached_train"],
            epochs=self.khp["epochs"],
            callbacks=[
                EarlyStopping(
                    monitor="loss",  # val dataset not parallel in tfrs
                    patience=self.khp["patience"],
                    restore_best_weights=True
                ),
                TensorBoard(
                    log_dir=self.paths["logs"],
                    histogram_freq=1,
                )
            ]
        )

        # get the optimal hyperparameters and model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.best_model = tuner.get_best_models(num_models=1)[0]

        # log
        logging.info(
            "best hyperparameters: %s",
            {h: best_hps.get(h) for h in best_hps}
        )
        logging.info('run tuner function completed')

    def evaluate_model(self):
        """ Evaluate the model. """

        self.best_model.evaluate(self.data["cached_test"], return_dict=True)

    def save_weights_and_ranking(self):
        """ Save the model embeddings and ranking model. """

        self.best_model.save_weights(self.paths["weights"], save_format="tf")
        self.best_model.ranking_model.save(self.paths["ranking_model"])

    def build_model(self):
        """ Convert hyperparameters; train, evaluate, and save the model. """

        self.convert_hyperparameters()
        self.tune_model()
        self.evaluate_model()
        self.save_weights_and_ranking()
