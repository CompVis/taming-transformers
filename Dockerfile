FROM armswdev/pytorch-arm-neoverse:r22.02-torch-1.10.0-openblas as base
WORKDIR /home/taming-transformers

COPY pyproject.toml pyproject.toml

ARG APP_MODE=development

ENV APP_MODE=${APP_MODE} \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.1.12

RUN pip install "poetry==$POETRY_VERSION"


# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi
