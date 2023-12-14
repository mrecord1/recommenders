"""
This script orchestrates the training and evaluation of a tensorflow
recommenders two tower model with both ranking and retrieval. The query and
candidate data are TCG cards, where the query data represents an instance of
user, deck, and included card, which implies a positive rating of the card.
The candidate data represents all cards in the TCG. The concept of ranking is
engineered by calculating the relative frequency of card appearances.
The architecture is flexible for additional features and optimization.
The resulting model is still considered exploratory, but it is very effective.
"""

import logging
from typing import TextIO

from modules.yaml_readers import ConfigYamlReader, SchemaYamlReader
from modules.pandas_reader import PandasReader
from modules.pandas_pipeline import PandasPipeline
from modules.hyperparams import ModelTuner


def main(config_file: TextIO):
    """
    Orchestrate the end-to-end training process.

    Args:
        config_file: yaml config file reference.
    """

    # define storage for data files (vocabs and tf datasets)
    data = {}

    # get configuration (hyperparameters and file paths)
    config_reader = ConfigYamlReader(filepath=config_file)
    hp, paths = config_reader.process_file()

    # get schemas for query and candidate data
    schema_reader = SchemaYamlReader(filepath=paths["schemas"])
    schemas = schema_reader.process_file()

    # read query data, then store the dataframe, num queries, and vocabs
    read_q = PandasReader(schemas=schemas, schema_type="query")
    results_q = read_q.process(fileinfo=paths["queries"])
    df_q, hp["fixed"]["n_q"], data["vocabs_q"] = results_q

    # read candidate data, then store the dataframe, num candidates, and vocabs
    read_c = PandasReader(schemas=schemas, schema_type="candidate")
    results_c = read_c.process(fileinfo=paths["candidates"])
    df_c, hp["fixed"]["n_c"], data["vocabs_c"] = results_c

    # run the full pipeline with schema and hyperparameters to get cached tf data
    train_pipe = PandasPipeline(schema=read_q.schema, hp=hp)
    data["cached_train"], data["cached_test"] = train_pipe.run_pipeline(df_q)

    # get index data from candidate data leveraging only the conversion step
    index_pipe = PandasPipeline(schema=read_c.schema, hp=hp)
    data["index_data"] = index_pipe.convert_to_tf(df_c, dataset_type="index")

    # tune, train, and eval, then save weights and ranking submodel
    tuner = ModelTuner(data=data, hp=hp, paths=paths)
    tuner.build_model()

    # predictions made separately from saved submodels

    # log completion message
    logging.info('main function completed')


if __name__ == "__main__":

    # set up logging
    logging.basicConfig(
        filename='logs/app.log',
        level=logging.DEBUG,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
    )

    # run main function
    main("config.yaml")
