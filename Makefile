.PHONY: setup test data api demo

setup:
	python -m pip install -r requirements.txt

data:
	python -m src.data.ingestion --config config/config.yaml --data-dir data/raw

test:
	pytest -q

api:
	uvicorn demo.api:app --reload

demo:
	streamlit run demo/app.py
