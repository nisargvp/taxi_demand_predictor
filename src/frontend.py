import zipfile 
from datetime import datetime, timedelta

import zipfile
from datetime import datetime

import requests
import numpy as np
import pandas as pd

# plotting libraries
import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import (
    load_predictions_from_store,
    load_batch_of_features_from_store
)
from src.paths import DATA_DIR
from src.plot import plot_one_sample

st.set_page_config(layout="wide")

# to access local host type into Simple Browser: http://localhost:8501

# title
# current_date = datetime.strptime('2023-01-05 12:00:00', '%Y-%m-%d %H:%M:%S')
current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
st.title(f'Taxi demand prediction 🚕')
st.header(f'{current_date} UTC')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 6


def load_shape_data_file() -> gpd.geodataframe.GeoDataFrame:
    """
    Fetches remote file with shape data, that we later use to plot the
    different pickup_location_ids on the map of NYC.

    Raises:
        Exception: when we cannot connect to the external server where
        the file is.

    Returns:
        GeoDataFrame: columns -> (OBJECTID	Shape_Leng	Shape_Area	zone	LocationID	borough	geometry)
    """
    # download zip file
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    response = requests.get(URL)
    path = DATA_DIR / f'taxi_zones.zip'
    if response.status_code == 200:
        open(path, "wb").write(response.content)
    else:
        raise Exception(f'{URL} is not available')

    # unzip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')

    # load and return shape file
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')


@st.cache_data
def _load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """Wrapped version of src.inference.load_batch_of_features_from_store, so
    we can add Streamlit caching

    Args:
        current_date (datetime): _description_

    Returns:
        pd.DataFrame: n_features + 2 columns:
            - `rides_previous_N_hour`
            - `rides_previous_{N-1}_hour`
            - ...
            - `rides_previous_1_hour`
            - `pickup_hour`
            - `pickup_location_id`
            - `pickup_ts`
    """
    return load_batch_of_features_from_store(current_date)


@st.cache_data
def _load_predictions_from_store(
    from_pickup_hour: datetime,
    to_pickup_hour: datetime
    ) -> pd.DataFrame:
    """
    Wrapped version of src.inference.load_predictions_from_store, so we
    can add Streamlit caching

    Args:
        from_pickup_hour (datetime): min datetime (rounded hour) for which we want to get
        predictions

        to_pickup_hour (datetime): max datetime (rounded hour) for which we want to get
        predictions

    Returns:
        pd.DataFrame: 3 columns:
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
    """
    return load_predictions_from_store(from_pickup_hour, to_pickup_hour)


with st.spinner(text="Downloading shape file to plot taxi zones"):
    geo_df = load_shape_data_file()
    st.sidebar.write('✅ Shape file was downloaded ')
    progress_bar.progress(1/N_STEPS)
    
# with st.spinner(text="Fetching model predictions from the store"):
#     predictions_df = _load_predictions_from_store(
#         from_pickup_hour=current_date - timedelta(hours=1),
#         to_pickup_hour=current_date
#     )
#     st.sidebar.write('✅ Model predictions arrived')
#     progress_bar.progress(2/N_STEPS)
    
# # Here we are checking the predictions for the current hour have already been computed
# # and are available
# next_hour_predictions_ready = \
#     False if predictions_df[predictions_df.pickup_hour == current_date].empty else True
# prev_hour_predictions_ready = \
#     False if predictions_df[predictions_df.pickup_hour == (current_date - timedelta(hours=1))].empty else True


with st.spinner(text="Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write('✅ Inference features fetched from the store')
    progress_bar.progress(2/N_STEPS)
    print(f'{features}')
    
with st.spinner(text="Loading ML model from the registry"):
    model = load_model_from_registry()
    st.sidebar.write('✅ Model loaded from the registry')
    progress_bar.progress(3/N_STEPS)
    
with st.spinner(text="Computing model predictions"):
    results = get_model_predictions(model, features)
    st.sidebar.write('✅ Model predictions computed')
    progress_bar.progress(4/N_STEPS)