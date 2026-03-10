# MLOps Mini Pipeline

This repository contains a small **MLOps learning project** where I built a simple end-to-end machine learning pipeline to understand how modern ML systems are developed and deployed.

The goal of this project is **not to build a production system**, but to practice important MLOps concepts such as experiment tracking, CI pipelines, containerization, and model deployment.

## What this project demonstrates

- Basic ML pipeline for training and evaluation
- Experiment tracking using **MLflow**
- CI pipeline using **GitHub Actions**
- Containerization using **Docker**
- Model serving using **FastAPI**
- Image publishing to **Docker Hub**
- Deployment of the container on **AWS EC2**

## Project Workflow

1. Data preprocessing and feature generation  
2. Model training and evaluation  
3. Experiment logging with MLflow  
4. CI pipeline runs tests and builds Docker image  
5. Docker image is pushed to Docker Hub  
6. The container can be deployed on AWS EC2 for inference

## Technologies Used

- Python
- FastAPI
- MLflow
- Docker
- GitHub Actions
- AWS EC2
- DVC (optional depending on your pipeline)

## Purpose

This project was created to learn practical **MLOps tools and workflows** used in industry. It helped me understand how machine learning models move from experimentation to deployment.
