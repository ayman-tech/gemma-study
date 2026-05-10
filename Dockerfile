FROM python:3.12-slim

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency definition files first
COPY pyproject.toml uv.lock ./

# Install dependencies (leverages Docker cache)
RUN uv sync --frozen

# Copy the rest of the application
COPY . .

# Set default entrypoint
ENTRYPOINT ["uv", "run", "main.py"]
