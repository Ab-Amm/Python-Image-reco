# Step 1: Use a Python 3.8 base image
FROM python:3.8-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file to the container
COPY requirements.txt /app/

# Step 4: Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the application code into the container
COPY . /app/

# Step 6: Expose the port
EXPOSE 5000

# Step 7: Run the application
CMD ["python", "app.py"]
