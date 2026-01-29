# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv for package management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv pip install --system -e .

# Copy application code
COPY app.py ./

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
