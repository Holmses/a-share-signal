FROM node:22-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json /frontend/
RUN npm ci
COPY frontend /frontend
RUN npm run build

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY --from=frontend-build /frontend/dist /opt/a-share-dashboard

RUN pip install --no-cache-dir .

WORKDIR /workspace

ENTRYPOINT ["ashare-signal"]
