.PHONY: install data train evaluate app test lint format clean

install:
	pip install -r requirements.txt

data:
	python scripts/train_all.py --data-only

train:
	python scripts/train_all.py

evaluate:
	python scripts/evaluate_all.py

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/ app/

format:
	black src/ tests/ scripts/ app/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
