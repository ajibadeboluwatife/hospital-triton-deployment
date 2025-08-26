#!/bin/bash
docker build -t hospital-triton .
docker run --gpus all -p8000:8000 -p8001:8001 -p8002:8002 hospital-triton
