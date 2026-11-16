# Agentic Vision Data Engine (AVDE)

**Agentic Vision Data Engine: A Natural Language Interface for End-to-End Vision Data Orchestration**

📄 [Read the paper (PDF)](Agentic_Vision_Data_Engine.pdf)

**Authors:** Abeyankar Giridharan\*, Sai Sneha Siddapura Venkataramappa\*, Rusheng Zhang, Rajanikant Patnaik Ananta, Derek Johnson, Daniel Bogdoll, Aryaman Rao, Gregory Stevens, Gaurav Pandey, Henry X. Liu

\*Equal contribution

University of Michigan Transportation Research Institute (Mcity) · Karlsruhe Institute of Technology · Texas A&M University

## Overview

AVDE adds an LLM-powered conversational layer on top of the [Mcity Data Engine](https://github.com/mcity/mcity_data_engine). Users describe computer vision workflows in plain language. The agent then configures and runs them, with no manual config file edits or dataset format conversion.

## Key Features

- **LLM-agnostic:** interchangeable backends for OpenAI, Anthropic Claude, Google Gemini, and Groq-hosted models
- **Model Context Protocol (MCP):** each Data Engine workflow is exposed as a callable tool, including auto-labeling, class mapping, anomaly detection, embedding extraction, zero-shot labeling, and ensemble selection
- **Unified data ingestion:** auto-detects COCO, YOLO, CVAT, Pascal VOC, image folders, and raw video, then converts to Voxel51
- **Workflow validation:** tracks conversation state and checks dependencies so workflows run in the correct order
- **Experiment tracking:** Weights & Biases integration for metrics, hyperparameters, and checkpoints
- **Cloud deployment:** self-service, per-user AWS instances (CloudFormation + EC2) with Redis rate limiting and scheduled teardown, ready in roughly 5 minutes

## Results

Evaluated on roadside fisheye object detection with YOLOv12 and Co-DETR:

- Setup and configuration time dropped from 10 to 40 minutes (manual) to roughly 3 minutes per task, with 1 to 2 rounds of dialog
- Training, fine-tuning, and evaluation all ran from natural-language instructions only

## Code

The AVDE implementation lives on the [`aws-agentic-dataengine`](https://github.com/mcity/mcity_data_engine/tree/aws-agentic-dataengine) branch.

## Acknowledgments

Supported by Mcity, University of Michigan Transportation Research Institute (UMTRI), with cloud computing resources provided in part by Amazon Web Services.
