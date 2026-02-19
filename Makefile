.PHONY: install test lint clean run dashboard docker-build docker-run docker-cli docker-up docker-down

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.cli $(ARGS)

dashboard:
	streamlit run src/dashboard/app.py

docker-build:
	docker build -t retail-analytics .

docker-run:
	docker run -p 8501:8501 -v $(PWD)/data:/app/data retail-analytics

docker-cli:
	docker run -v $(PWD)/data:/app/data -v $(PWD)/input:/app/input -v $(PWD)/output:/app/output \
		retail-analytics python -m src.cli $(ARGS)

docker-up:
	docker compose up -d

docker-down:
	docker compose down
