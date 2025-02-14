# Use the official PyTorch image with CUDA support (change the tag if needed)
FROM pytorch/pytorch

# Set a working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  python3-pip \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set the default command to run a PyTorch script (change main.py as needed)
CMD ["python", "train.py"]

