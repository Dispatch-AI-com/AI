# ---- Config ----
SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# 可在 CI 里覆盖：make test PYTHON=3.12
PYTHON ?= python3
CODE ?= .
TEST_DIR ?= tests            # 如果你确实放在 app/test，就写 TEST_DIR=app/test
MYPY_TARGET ?= $(CODE)/app
PYTEST_OPTS ?= -q --maxfail=1 --disable-warnings
COV ?= 1                    # 设为 0 关闭覆盖率
COV_OPTS := $(if $(filter $(COV),1),--cov=$(CODE) --cov-report=xml,)

# ---- Phonies ----
.PHONY: help sync lint lint-fix format test typecheck check-all

help:
	@echo "make sync        安装/同步依赖（uv sync）"
	@echo "make lint        Ruff 代码检查"
	@echo "make lint-fix    Ruff 自动修复"
	@echo "make format      Ruff 格式化"
	@echo "make typecheck   MyPy 类型检查"
	@echo "make test        Pytest 测试（含覆盖率，CI 默认开启）"
	@echo "make check-all   lint + typecheck + test"

sync:
	uv sync --frozen || uv sync   # 有 uv.lock 就严格锁定；没有则创建

lint:
	uv run ruff check $(CODE)

lint-fix:
	uv run ruff check --fix $(CODE)

format:
	uv run ruff format $(CODE)

typecheck:
	uv run mypy $(MYPY_TARGET)

test:
	uv run pytest $(TEST_DIR) $(PYTEST_OPTS) $(COV_OPTS)

check-all: lint typecheck test
	@echo "✅ All checks passed!"
