# Use the official Python 3.12 slim image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the API when the container starts
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]