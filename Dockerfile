FROM public.ecr.aws/lambda/python:3.10

# 1. Install System Dependencies & Build Tools
# Added: gcc-c++, make, cmake, unzip (Required to compile soxr and other libs)
RUN yum install -y \
    libsndfile \
    ffmpeg \
    git \
    gcc-c++ \
    make \
    cmake \
    unzip \
    && yum clean all

# 2. Install uv (Rest of the file remains the same...)
RUN pip install uv

# 3. Copy requirements
COPY requirements.txt .

# 4. Install PyTorch CPU-only
RUN uv pip install --system --no-cache \
    "torch==2.0.1+cpu" \
    "torchaudio==2.0.2+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Install librosa and requirements
RUN uv pip install --system --no-cache "soxr==0.3.7" -r requirements.txt
# 6. Set Numba Cache
ENV NUMBA_CACHE_DIR=/tmp 
COPY ./model.py ${LAMBDA_TASK_ROOT}/
COPY ./app.py ${LAMBDA_TASK_ROOT}/

# Copy model artifacts
COPY ./artifacts/ ${LAMBDA_TASK_ROOT}/artifacts/

# Set the CMD to your handler
CMD ["app.lambda_handler"]