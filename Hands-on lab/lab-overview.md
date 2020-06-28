# Machine Learning hands-on lab step-by-step

## Abstract and learning objectives

In this hands-on lab, you will use Azure Databricks in combination with Azure Machine Learning to build, train and deploy desired models. You will learn how to train a forecasting model against time-series data, without any code, by using automated machine learning, and how to score data in real-time using Spark Structure Streaming within Azure Databricks.  You will create a recurrent neural network (RNN) model using PyTorch in Azure Databricks that can be used to forecast against time-series data and train a Natural Language Processing (NLP) text classification model using Keras. You will also learn how to use MLflow for managing experiments run directly on the Azure Databricks cluster and how MLflow can seamlessly log metrics and training artifacts in your Azure Machine Learning workspace.

At the end of this lab, you will improve your ability to build solutions leveraging Azure Machine Learning and Azure Databricks.

## Overview

Trey Research Inc. delivers innovative solutions for manufacturers. They specialize in identifying and solving problems for manufacturers that can run the range from automating away mundane but time-intensive processes to delivering cutting edge approaches that provide new opportunities for their manufacturing clients.

Trey Research is looking to provide the next generation experience for connected car manufacturers by enabling them to utilize AI to decide when to pro-actively reach out to the customer through alerts delivered directly to the car's in-dash information and entertainment head unit. For their proof of concept (PoC), they would like to focus on two maintenance related scenarios.

In the first scenario, Trey Research recently instituted new regulations defining what parts are compliant or out of compliance. Rather than rely on their technicians to assess compliance, they would like to automatically assess the compliance based on component notes already entered by authorized technicians. Specifically, they are looking to leverage Deep Learning technologies with Natural Language Processing techniques to scan through vehicle specification documents to find compliance issues with new regulations. Then each car is evaluated for out compliance components.

In the second scenario, Trey Research would like to predict the likelihood of battery failure based on the telemetry stream of time series data that the car provides about how the battery performs when the car is started, how it is charging while running and how well it is holding its charge, among other factors. If they detect a battery failure is imminent within the next 30 days, they would like to send an alert.

Upon detection of an out of compliance component or a battery at risk of failure, they would like to be able to send an alert directly to the customer inviting them to schedule a service appointment to replace the part.

In building this PoC, Trey Research wants to understand how they might use machine learning or deep learning in both scenarios, and standardize the platform that would support the data processing, model management and inferencing aspects of each.

They are also interested to learn what new capabilities Azure provides that might help them to integrate with their existing investments in MLflow for managing machine learning experiments. Furthermore, they would also like to understand how Azure might help them to document and explain the models that are created to non-data scientists or might accelerate their time to creating production ready, performant models.

In this lab, you will use Azure Databricks in combination with Azure Machine Learning to build, train and deploy the desired models.

## Solution architecture

The following diagram summarizes the key components and processing steps in the lab.

![Vehicle battery telemetry is ingested by an IoT Hub or Event Hub. This data is stored in long term storage, Azure Storage. This data is used by Azure Databricks to train the model that is managed and registered via an Azure Machine Learning workspace. AutoML is also another option that can be used to register a machine learning model. These models are then used for stream data processing and batch data processing in Azure Databricks.](images/lab-architecture.png 'Solution Architecture')

In this lab, models are trained using both Azure Databricks (for deep learning with the PyTorch and Keras frameworks) and Azure Machine Learning compute (for automated machine learning using the user experience in the Azure Machine Learning studio). Models are registered with the Azure Machine Learning Workspace. The data used for model training is read from Azure Storage.

The scoring is performed using notebooks running within Azure Databricks notebooks, which show how to load and apply the respective models against the data provided.

## Requirements

1. Microsoft Azure subscription must be Pay-As-You-go or MSDN.

    - Trial subscriptions will not work. You will run into issues with Azure resource quota limits.

    - Subscriptions with access limited to a single resource group will not work. You will need the ability to deploy multiple resource groups.
    
### Contents

<!-- TOC -->

- [Machine Learning hands-on lab step-by-step](#machine-learning-hands-on-lab-step-by-step)
  - [Abstract and learning objectives](#abstract-and-learning-objectives)
  - [Overview](#overview)
  - [Solution architecture](#solution-architecture)
  - [Requirements](#requirements)
  - [Before the hands-on lab](#before-the-hands-on-lab)
  - [Exercise 1: Creating a forecast model using automated machine learning](#exercise-1-creating-a-forecast-model-using-automated-machine-learning)
    - [Task 1: Create an automated machine learning experiment](#task-1-create-an-automated-machine-learning-experiment)
    - [Task 2: Review the experiment run results](#task-2-review-the-experiment-run-results)
    - [Task 3: Deploy the Best Model](#task-3-deploy-the-best-model)
  - [Exercise 2: Creating a deep learning model (RNN) for time series data](#exercise-2-creating-a-deep-learning-model-rnn-for-time-series-data)
    - [Task 1: Create the model using a notebook](#task-1-create-the-model-using-a-notebook)
  - [Exercise 3: Using a forecast model for scoring of streaming telemetry](#exercise-3-using-a-forecast-model-for-scoring-of-streaming-telemetry)
    - [Task 1: Create the streaming job using a notebook](#task-1-create-the-streaming-job-using-a-notebook)
  - [Exercise 4: Creating, training and tracking a deep learning text classification model with Azure Databricks, MLflow and Azure Machine Learning](#exercise-4-creating-training-and-tracking-a-deep-learning-text-classification-model-with-azure-databricks-mlflow-and-azure-machine-learning)
    - [Task 1: Create, train and track the classification model using a notebook](#task-1-create-train-and-track-the-classification-model-using-a-notebook)
    - [Task 2: Review model performance metrics and training artifacts in Azure Machine Learning workspace](#task-2-review-model-performance-metrics-and-training-artifacts-in-azure-machine-learning-workspace)
  - [After the hands-on lab](#after-the-hands-on-lab)
    - [Task 1: Clean up lab resources](#task-1-clean-up-lab-resources)

<!-- /TOC -->
