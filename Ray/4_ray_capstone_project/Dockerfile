# Base Linux image with Miniconda already installed.
FROM continuumio/miniconda3:26.1.1-1

# All project files inside the container will live here.
WORKDIR /workspace

# Compose builds from the Ray part root, so this copies the part-local env spec.
COPY environment.yml /tmp/environment.yml

# Strip the editable-install line — the project is installed separately below.
RUN sed -i '/-e \./d' /tmp/environment.yml

# Create a full conda environment inside the image.
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Copy project files and install the package in editable mode.
COPY . /workspace
RUN conda run -n 22971-ray-capstone pip install --no-cache-dir -e .

# Make the conda env the default for all subsequent commands.
ENV PATH=/opt/conda/envs/22971-ray-capstone/bin:$PATH
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command
CMD ["python", "--version"]
