FROM nvcr.io/nvidia/tritonserver:23.09-py3

# Copy hospital model repo into container
COPY ./models /models

# Expose Triton ports
EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/models"]
