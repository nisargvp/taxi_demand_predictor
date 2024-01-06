from datetime import datetime, timedelta
from argparse import ArgumentParser

import pandas as pd

import src.config as config
from src.logger import get_logger
from src.config import FEATURE_GROUP_PREDICTIONS_METADATA, FEATURE_GROUP_METADATA
from src.feature_store_api import get_or_create_feature_group, get_feature_store

logger = get_logger()


def load_predictions_and_actual_values_from_store(
    from_date: datetime, 
    to_date: datetime
) -> pd.DataFrame:
    """Fetches model predictions and actual values from 
    `from_date` to `to_date` from the Feature Store and returns a DataFrame.

    Args:
        from_date (datetime): min datetime for which we want predictions and 
        actual values
        to_date (datetime): max datetime for which we want predictions and
        actual values

    Returns:
        pd.DataFrame: 4 columns: 
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
            - `rides`
    """
    # 2 feature groups we need to merge
    predictions_fg = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA)
    actuals_fg = get_or_create_feature_group(FEATURE_GROUP_METADATA)
    
    # query to join the 2 feature groups by `pickup_hour` and `pickup_location_id`
    from_ts = int(from_date.timestamp() * 1000)
    to_ts = int(to_date.timestamp() * 1000)
    query = predictions_fg.select_all() \
        .join(actuals_fg.select(['pickup_location_id', 'pickup_ts', 'rides']),
              on=['pickup_ts', 'pickup_location_id'], prefix=None) \
        .filter(predictions_fg.pickup_ts >= from_ts) \
        .filter(predictions_fg.pickup_ts <= to_ts) 
        
    # breakpoint() 
    # after (Pdb): l for list of current location in the source code, n for next line, c for continue and name of the variable to print its value
    
    # create the feature view `config.FEATURE_VIEW_PREDICTIONS_METADATA` if it doest not 
    # exist yet
    feature_store = get_feature_store() 
    try:
        # create feature view as it does not exist yet
        feature_store.create_feature_view(
            name=config.MONITORING_FV_NAME,
            version=config.MONITORING_FV_VERSION,
            query=query
        )
    except:
        logger.info('Feture view already existed. Skip creation.')
    
    # feature view
    monitoring_fv = feature_store.get_feature_view(
        name=config.MONITORING_FV_NAME,
        version=config.MONITORING_FV_VERSION
    )
    
    # featch data from the feature view
    # fetch predicted and actual values plus/minus 30 days to make sure we have enough data
    monitoring_df = monitoring_fv.get_batch_data(
        start_time=from_date - timedelta(days=30),
        end_time=to_date + timedelta(days=30)
    )
    
    # filter data to the time period we are interested in
    pickup_ts_from = int(from_date.timestamp() * 1000)
    pickup_ts_to = int(to_date.timestamp() * 1000)
    monitoring_df = monitoring_df[monitoring_df.pickup_ts.between(pickup_ts_from, pickup_ts_to)]
    
    return monitoring_df

# if __name__ == '__main__':
    
#     # parse command line arguments
#     parser = ArgumentParser()
#     parser.add_argument('--from_date', 
#                         type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
#                         help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
#     parser.add_argument('--to_date', 
#                         type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
#                         help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
#     args = parser.parse_args()
    
    
#     monitoring_df = load_predictions_and_actual_values_from_store()
    
