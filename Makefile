.PHONY: init features training inference frontend monitoring

# downloads Poetry and installs all dependencies from pyproject.toml
init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install

# generates new batch of features and stores them in the feature store
features:
	poetry run python scripts/feature_pipeline.py

# trains a new model and stores it in the model registry
training:
	poetry run python scripts/training_pipeline.py

# generates predictions and stores them in the feature store
inference:
	poetry run python scripts/inference_pipeline.py

# backfills the feature store with historical data
backfill:
	poetry run python scripts/backfill_feature_group.py

# starts the Streamlit app
frontend-app:
	poetry run streamlit run src/frontend.py

monitoring-app:
	poetry run streamlit run src/frontend_monitoring.py