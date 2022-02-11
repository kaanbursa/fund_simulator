import pandas as pd
from datetime import datetime, timedelta
import os
import wandb
from data.preprocessing import get_price
from data.Clustering import UnsupervisedLearning


STOCKS_FILE_PATH = './data/all_stocks.csv'

def get_clustered_stocks(df, from_date, to_date, number_of_clusters=10):

    lst1 = df[(df.Date >= pd.to_datetime(from_date)) & (
                df.Date < pd.to_datetime(from_date + timedelta(days=4)))].ticker.unique()
    lst2 = df[(df.Date >= pd.to_datetime(to_date - timedelta(days=4))) & (
            df.Date < pd.to_datetime(to_date ))].ticker.unique()
    existing_stocks = [value for value in lst1 if value in lst2]
    training_df = df[df['ticker'].isin(existing_stocks)]

    training_df = training_df[
        (training_df.Date >= pd.to_datetime(from_date)) & (training_df.Date < pd.to_datetime(to_date))]
    # For clustering
    training_df.Date = training_df.Date.apply(lambda x: x.strftime('%Y-%m-%d'))

    UL = UnsupervisedLearning(training_df)


    cluster = UL.kmeans_cluster_stocks(number_of_clusters)
    return cluster, training_df

def prepare_stocks(stock_path):
    if os.path.exists(STOCKS_FILE_PATH):
        stock_df = pd.read_csv(STOCKS_FILE_PATH)

        symbols = stock_df.Symbol.unique()
    else:
        raise KeyError
    return symbols

def load_all_stocks(start_date, all_stocks_path):
    symbols = prepare_stocks(all_stocks_path)
    df = pd.DataFrame()
    if os.path.exists(all_stocks_path):
        df = pd.read_csv(all_stocks_path)
        df.Date = pd.to_datetime(df.Date)
    else:
        for symbol in symbols:
            stock = get_price(symbol, start_date)
            df = pd.concat([df, stock])
        df.to_csv(all_stocks_path)
        df.Date = pd.to_datetime(df.Date)

    return df

def log_trade(info):

    with wandb.init('fund-simulation', entity='kaanb') as run:
        run.log(info)


def load_and_log(dataset, description, indicator_list, stock_list, name='RLFund') -> None:
    """
    Wandb helper functiton for dataset versioning
    :param dataset:
    :param name: name for the artifact usualy is the same
    :param description: description to write
    :param indicator_list: to find which list is used in this model
    :param stock_list: unique stock list
    :return: None
    """
    with wandb.init('fund-simulation', entity='kaanb') as run:
        raw_data = wandb.Artifact(
            name, type="dataset",
            description=description,
            metadata={
                'sizes':len(dataset),
                'indicator_list': indicator_list,
                'stock_list':stock_list
            }
        )

        run.log_artifact(raw_data)

def preprocess_and_log(name, steps):
    with wandb.init(project='fund-simulation', job_type="preprocess-data") as run:
        processed_data = wandb.Artifact(
            name, type='dataset',
            description="Preprocessing of stock data",
            metadata=steps
        )
        run.log_artifact(processed_data)

def build_and_log_model(model, model_alias, model_name, config):
    with wandb.init(project='fund-simulation', job_type='initialize', config=config) as run:
        config = wandb.config
        model_artifact = wandb.Artifact(
            model_name, type="model",
            description="Reinforcement learning agent",
            metadata=dict(config)

        )

        # save stable baselines model and upload
        model.save(f'{model_alias}.zip')
        model_artifact.add_file(f'{model_alias}.zip')
        wandb.save(f'{model_alias}.zip')

        run.log_artifact(model_artifact)



def visualize_validation_reward(df):
    pass