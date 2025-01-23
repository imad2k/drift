# filepath: /Users/Imad/Documents/Dev/Drift/backend/services/Dockerfile
# Use the official Python image from the Docker Hub
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Copy the .env file into the container
COPY .env . 

# Set the environment variable for Flask
ENV FLASK_APP=main.py

# Expose the port the app runs on
EXPOSE 5001

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]