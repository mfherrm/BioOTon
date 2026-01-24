FROM rayproject/ray:2.37.0-py310-gpu

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy config and install dependencies
COPY pyproject.toml .
RUN uv pip install --system --no-cache .

COPY . .

ENTRYPOINT ["python", "main.py"]
