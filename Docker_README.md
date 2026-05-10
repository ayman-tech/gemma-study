# Docker Usage Guide: Gemma-Study

This project has been fully containerized, and the image is available on Docker Hub as `55rahul55/gemma-study`.

## Prerequisites
- [Docker](https://www.docker.com/get-started) installed on your machine.

## How to Pull the Image
To pull the latest version of the image from Docker Hub, run:

```bash
docker pull 55rahul55/gemma-study:latest
```

## How to Run the Image
The Docker container is configured to automatically run the `main.py` benchmark script as its entrypoint. You can pass the script's arguments directly to the `docker run` command.

### 1. Run with Default Settings
By default, this will run the **Gemma** model benchmark on the **CPU**:

```bash
docker run -it 55rahul55/gemma-study
```

### 2. Run the ViT Benchmark
To run the Vision Transformer (ViT) model benchmark on the CPU:

```bash
docker run -it 55rahul55/gemma-study --model vit --device cpu
```

### 3. Run Both Models
To benchmark both models back-to-back:

```bash
docker run -it 55rahul55/gemma-study --model both
```

### 4. Run with GPU Support
If you have an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, you can use your GPU to accelerate the benchmarks by adding the `--gpus all` flag:

```bash
docker run --gpus all -it 55rahul55/gemma-study --model gemma --device gpu --quantize 4bit
```

### 5. Accessing the Interactive Shell
If you want to explore the container's environment or run commands manually without immediately triggering the benchmark script, you can override the entrypoint to launch a bash shell:

```bash
docker run -it --entrypoint /bin/bash 55rahul55/gemma-study
```
