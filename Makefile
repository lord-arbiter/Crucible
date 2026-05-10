.PHONY: help install dev-install lint test smoke precache serve frontend deploy docker-build docker-run clean

help:
	@echo "Crucible — common commands"
	@echo "  install         install runtime deps"
	@echo "  dev-install     install runtime + dev deps"
	@echo "  lint            ruff check"
	@echo "  test            pytest"
	@echo "  smoke           one-episode end-to-end smoke test against vLLM"
	@echo "  precache REPO=  pre-cache scoring results for a dataset"
	@echo "  serve           run FastAPI orchestrator on :8000"
	@echo "  frontend        run Gradio frontend on :7860"
	@echo "  docker-build    build the MI300X GPU image"
	@echo "  docker-run      run the GPU image (requires --device=/dev/kfd etc.)"
	@echo "  deploy          push frontend to a HuggingFace Space"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

lint:
	ruff check src frontend scripts tests

test:
	pytest -q

smoke:
	python scripts/one_shot_test.py --repo $${REPO:-lerobot/aloha_static_cups_open} --critic visual

precache:
	@if [ -z "$$REPO" ]; then echo "Set REPO=<lerobot/...>"; exit 1; fi
	python scripts/precache_demo.py --repo $$REPO --episodes $${EPISODES:-25}

serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

frontend:
	CRUCIBLE_API_BASE=$${CRUCIBLE_API_BASE:-http://localhost:8000} python frontend/app.py

docker-build:
	docker build -f docker/Dockerfile.gpu -t crucible:gpu .

docker-run:
	docker run --rm -it \
		--device=/dev/kfd --device=/dev/dri \
		--security-opt seccomp=unconfined \
		--group-add video \
		-p 8000:8000 -p 8001:8001 \
		-v $$HOME/.cache/huggingface:/root/.cache/huggingface \
		crucible:gpu

deploy:
	HF_USER=$${HF_USER:?set HF_USER} SPACE_NAME=$${SPACE_NAME:-crucible} ./scripts/deploy_space.sh

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
