![Microsoft Cloud Workshops](https://github.com/Microsoft/MCW-Template-Cloud-Workshop/raw/master/Media/ms-cloud-workshop.png "Microsoft Cloud Workshops")

<div class="MCWHeader1">
Machine Learning
</div>

<div class="MCWHeader2">
Hands-on lab step-by-step
</div>

<div class="MCWHeader3">
November 2019
</div>


Information in this document, including URL and other Internet Web site references, is subject to change without notice. Unless otherwise noted, the example companies, organizations, products, domain names, e-mail addresses, logos, people, places, and events depicted herein are fictitious, and no association with any real company, organization, product, domain name, e-mail address, logo, person, place or event is intended or should be inferred. Complying with all applicable copyright laws is the responsibility of the user. Without limiting the rights under copyright, no part of this document may be reproduced, stored in or introduced into a retrieval system, or transmitted in any form or by any means (electronic, mechanical, photocopying, recording, or otherwise), or for any purpose, without the express written permission of Microsoft Corporation.

Microsoft may have patents, patent applications, trademarks, copyrights, or other intellectual property rights covering subject matter in this document. Except as expressly provided in any written license agreement from Microsoft, the furnishing of this document does not give you any license to these patents, trademarks, copyrights, or other intellectual property.

The names of manufacturers, products, or URLs are provided for informational purposes only and Microsoft makes no representations and warranties, either expressed, implied, or statutory, regarding these manufacturers or the use of the products with any Microsoft technologies. The inclusion of a manufacturer or product does not imply endorsement of Microsoft of the manufacturer or product. Links may be provided to third party sites. Such sites are not under the control of Microsoft and Microsoft is not responsible for the contents of any linked site or any link contained in a linked site, or any changes or updates to such sites. Microsoft is not responsible for webcasting or any other form of transmission received from any linked site. Microsoft is providing these links to you only as a convenience, and the inclusion of any link does not imply endorsement of Microsoft of the site or the products contained therein.

© 2019 Microsoft Corporation. All rights reserved.

Microsoft and the trademarks listed at <https://www.microsoft.com/en-us/legal/intellectualproperty/Trademarks/Usage/General.aspx> are trademarks of the Microsoft group of companies. All other trademarks are property of their respective owners.

**Contents** 

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
  - [Exercise 2: Creating a deep learning model (RNN) for time series data and registering the model](#exercise-2-creating-a-deep-learning-model-rnn-for-time-series-data-and-registering-the-model)
    - [Task 1: Create the model using a notebook](#task-1-create-the-model-using-a-notebook)
  - [Exercise 3: Using a forecast model for scoring of streaming telemetry](#exercise-3-using-a-forecast-model-for-scoring-of-streaming-telemetry)
    - [Task 1: Create the streaming job using a notebook](#task-1-create-the-streaming-job-using-a-notebook)
  - [Exercise 4: Creating a deep learning text classification model](#exercise-4-creating-a-deep-learning-text-classification-model)
    - [Task 1: Create the classification model using a notebook](#task-1-create-the-classification-model-using-a-notebook)
  - [After the hands-on lab](#after-the-hands-on-lab)
    - [Task 1: Clean up lab resources](#task-1-clean-up-lab-resources)

<!-- /TOC -->

# Machine Learning hands-on lab step-by-step 

## Abstract and learning objectives 

In this hands-on lab, you will use Azure Databricks in combination with Azure Machine Learning to build, train and deploy desired models. You will learn how to train a forecasting model against time-series data, without any code, by using automated machine learning, and how to score data in real-time using Spark Structure Streaming within Azure Databricks.  You will create a recurrent neural network (RNN) model using PyTorch in Azure Databricks that can be used to forecast against time-series data and train a Natural Language Processing (NLP) text classification model using Keras.

At the end of this lab, you will improve your ability to build solutions leveraging Azure Machine Learning and Azure Databricks.

## Overview

Trey Research Inc. delivers innovative solutions for manufacturers. They specialize in identifying and solving problems for manufacturers that can run the range from automating away mundane but time-intensive processes to delivering cutting edge approaches that provide new opportunities for their manufacturing clients.

Trey Research is looking to provide the next generation experience for connected car manufacturers by enabling them to utilize AI to decide when to pro-actively reach out to the customer through alerts delivered directly to the car's in-dash information and entertainment head unit. For their proof of concept (PoC), they would like to focus on two maintenance related scenarios.

In the first scenario, Trey Research recently instituted new regulations defining what parts are compliant or out of compliance. Rather than rely on their technicians to assess compliance, they would like to automatically assess the compliance based on component notes already entered by authorized technicians. Specifically, they are looking to leverage Deep Learning technologies with Natural Language Processing techniques to scan through vehicle specification documents to find compliance issues with new regulations. Then each car is evaluated for out compliance components.

In the second scenario, Trey Research would like to predict the likelihood of battery failure based on the telemetry stream of time series data that the car provides about how the battery performs when the car is started, how it is charging while running and how well it is holding its charge, among other factors. If they detect a battery failure is imminent within the next 30 days, they would like to send an alert.

Upon detection of an out of compliance component or a battery at risk of failure, they would like to be able to send an alert directly to the customer inviting them to schedule a service appointment to replace the part.

In building this PoC, Trey Research wants to understand how they might use machine learning or deep learning in both scenarios, and standardize the platform that would support the data processing, model management and inferencing aspects of each.

They are also interested to learn what new capabilities Azure provides that might help them to document and explain the models that are created to non-data scientists or might accelerate their time to creating production ready, performant models.

In this lab, you will use Azure Databricks in combination with Azure Machine Learning to build, train and deploy the desired models.

## Solution architecture

The following diagram summarizes the key components and processing steps in the lab.

![Overall solution diagram, described in the text that follows.](images/lab-architecture.png 'Solution Architecture')

In this lab, models are trained using both Azure Databricks (for deep learning with the PyTorch and Keras frameworks) and Azure Machine Learning compute (for automated machine learning using the user experience in the Azure Machine Learning studio). Models are registered with the Azure Machine Learning Workspace. The data used for model training is read from Azure Storage.

The scoring is performed using notebooks running within Azure notebooks, which show how to load and apply the respective models against the data provided.

## Requirements

1. Microsoft Azure subscription must be pay-as-you-go or MSDN

    - Trial subscriptions will not work. You will run into issues with Azure resource quota limits.

    - Subscriptions with access limited to a single resource group will not work. You will need the ability to deploy multiple resource groups.

## Before the hands-on lab

Refer to the Before the hands-on lab setup guide manual before continuing to the lab exercises.

## Exercise 1: Creating a forecast model using automated machine learning

Duration: 30 minutes

In this exercise, you will create a model that predicts battery failure from time-series data using the visual interface to automated machine learning in an Azure Machine Learning workspace.

### Task 1: Create an automated machine learning experiment

1. Navigate to your Azure Machine Learning workspace in the Azure Portal. Select **Launch the new Azure Machine Learning studio**.

    ![Navigate to the Azure Machine Learning workspace in the Azure Portal, open the Azure Machine Learning studio](images/automl-open-studio.png 'Open Azure Machine Learning studio')

    > **Note**: Alternatively, you can sign-in directly to the [Azure Machine Learning studio portal](https://ml.azure.com).

2. Select **Automated ML** in the left navigation bar.

    ![Open the Automated ML section in the Azure Machine Learning studio](images/automl-open.png 'Open Automated ML section')

3. Select **+ New automated ML run**.

    ![Automated machine learning section in Azure Machine Learning studio. The image highlights the "New automated ML run" button.](./images/automl-new-run.png 'Create new automated ML run')

4. Select **+ Create dataset** to start uploading your training data. Select the `From local files` option.

    ![Create a new dataset to be used by the automated ML run.](images/automl-create-dataset-01.png 'Create dataset for automated ML run')

5. In the `Basic info` section provide **Name** as `daily-battery-time-series` and select **Next**.

    ![Provide name for the dataset in the basic info section.](images/automl-create-dataset-02a.png 'Basic info: Name')

6. Select **Browse** and upload the `daily-battery-time-series-v2.csv` file you downloaded in preparation of this lab and then select **Next**.

    ![Upload the file containing training data in the Create Dataset wizard.](images/automl-create-dataset-02b.png 'Upload training data file')

7. On the Settings and preview panel, set the **column headers** drop down to `All files have same headers`, and then select **Next**.

8. Review the training data schema. Toggle the `Include` switch next to the column name to exclude the `Path`, `Column1`, `Number_Of_Trips`, `Lifetime_Cycles_Used` and `Battery_Rated_Cycles` columns. Select **Next**.

    ![The schema section shows both selected and ignored features from the uploaded dataset.](images/automl-create-dataset-03.png 'Select Features')

9. Review the dataset details in the `Confirm details` section and select **Create**.

10. Select the `daily-battery-time-series` dataset and then select **Next**.

    ![Select the newly created dataset and continue the creation of the automated machine learning run](images/automl-create-dataset-04.png 'Select newly created dataset')

11. Provide the experiment name: `Battery-Cycles` and select `Daily_Cycles_Used` as target column. Select **Create a new compute**.

    ![The experiment name is "Battery-Cycles", the target column is "Daily_Cycles_Used", and "Create a new compute" button is highlighted.](images/automl-create-dataset-05.png 'Create New Experiment details')

12. For the new compute, provide the following values and then select **Create**:

    - Compute name: `auto-ml-compute`
    - Select Virtual Machine size: `STANDARD_DS11_V2`
    - Minimum number of nodes: `1`
    - Maximum number of nodes: `1`

    ![Create a New Compute dialog shows the values for various fields, such as Compute name.](images/automl-create-compute.png 'Create a New Compute')

    The creation of the new compute may take several minutes. Once the process is completed, select **Next** in the `Configure run` section.

13. Select the `Time series forecasting` task type and then select `Date` as the time column and `Battery_ID` as group by column. Select **View additional configuration settings**.

    ![Configure the automated machine learning task type as a time series forecasting task with "Date" as the time column and "Battery_ID" as group by column.](images/automl-configure-task-01.png 'Configure time series forecasting task')

14. For the automated machine learning run additional configurations, provide the following values and then select **Save**:

    - Primary metric: `Normalized root mean squared error`

    - Forecast Horizon: enter `30`. This refers to forecasting out up to 30 days.

    - Training job time (hours) (in the `Exit criterion` section): enter `1`. This is the lowest value currently accepted.

    - Metric score threshold: enter `0.1355`. When this threshold value will be reached for an iteration metric the training job will terminate.

    ![The automated machine learning run additional configurations dialog shows values for various fields such as "Primary metric", "Forecast horizon" etc...](images/automl-configure-task-02.png 'Configure automated machine learning run additional configurations')

15. Select **Finish** to start the new automated machine learning run.

    > **Note**: The experiment should run for up to 10 minutes. If the run time exceeds 15 minutes, cancel the run and start a new one (steps 3, 10, 11, 12, 13, 14, and 15). Make sure you provide a lower value for `Metric score threshold` in step 14.

### Task 2: Review the experiment run results

1. Once the experiment completes, select `Details` to examine the details of the run containing information about the recommended model and the run summary.

   ![Run details shows recommended model and run summary.](images/automl-review-run-01.png 'Run details - recommended model and summary')

2. Select `Models` to see a table view of different iterations and select the iteration with the best `Normalized root mean square error` score. Note that the normalized root mean square error measures the error between the predicted value and actual value. In this case, the model with the lowest normalized root mean square error is the best model. Note that Azure Machine Learning Python SDK updates over time and gives you the best performing model at the time you run the experiment. Thus, it is possible that the best model you observe can be different than the one shown below.

    ![Run models shows a table view of iteration details and corresponding primary metric values.](images/automl-review-run-02.png 'Run Details - Models with their associated primary metric values')

### Task 3: Deploy the Best Model

1. Return to the details of your experiment run and select **Deploy best model**.

    ![The "Deploy best model" for an automated machine learning experiment run.](images/automl-deploy-best-model-01.png 'Deploy best model')

2. Provide the `Name`, `Description` and `Compute type`, and then select **Deploy**:

    - Name: **battery-cycles**

    - Description: **The best AutoML model to predict battery cycles.**

    - Compute type: Select `ACI`.

    ![The Deploy a model dialog that shows values name, description, and compute type.](images/automl-deploy-best-model-02.png 'Deploy the best model')

3. The model deployment process will register the model, create the deployment image, and deploy it as a scoring webservice in an Azure Container Instance (ACI). To view the deployed model, from Azure Machine Learning studio select **Endpoints** and then select the `Real-time endpoints` tab.

   ![Viewing the list of deployed model endpoints in the Azure Machine Learning workspace](images/automl-deploy-best-model-03.png 'Deployed model endpoints')

   > **Note**: The `battery-cycles` endpoint will show up in a matter of seconds but the actual deployment takes several minutes. You can check the deployment state of the endpoint by selecting it and then selecting the `Details` tab. A successful de deployment will have a state of `Healthy`.

4. If you see your model deployed in the above list, you are now ready to continue on to the next exercise.

## Exercise 2: Creating a deep learning model (RNN) for time series data and registering the model

Duration: 45 minutes

### Task 1: Create the model using a notebook

1. Browse to your Azure Databricks Workspace and open `AI with Databricks and AML \ Deep Learning with Time Series`. This is the notebook you will step through executing in this lab.

2. Follow the instructions within the notebook to complete the lab.

## Exercise 3: Using a forecast model for scoring of streaming telemetry

Duration: 45 minutes

In this exercise, you will apply the forecast model to a Spark streaming job in order to make predictions against streaming data.

### Task 1: Create the streaming job using a notebook

1. Browse to your Azure Databricks Workspace and navigate to `AI with Databricks and AML \ Stream Scoring`. This is the notebook you will step through executing in this lab.

2. Follow the instructions within the notebook to complete the lab.

## Exercise 4: Creating a deep learning text classification model

Duration: 45 minutes

In this exercise, you create a model for classifying component text as compliant or non-compliant.

### Task 1: Create the classification model using a notebook

1. Browse to your Azure Databricks Workspace and navigate to `AI with Databricks and AML \ Deep Learning with Text`. This is the notebook you will step through executing in this lab.

2. Follow the instructions within the notebook to complete the lab.

## After the hands-on lab

Duration: 5 minutes

To avoid unexpected charges, it is recommended that you clean up all of your lab resources when you complete the lab.

### Task 1: Clean up lab resources

1. Navigate to the Azure Portal and locate the `MCW-AI-Lab` Resource Group you created for this lab.

2. Select **Delete resource group** from the command bar.

    ![Screenshot of the Delete resource group button.](images/cleanup-delete-resource-group.png 'Delete resource group button')

3. In the confirmation dialog that appears, enter the name of the resource group and select **Delete**.

4. Wait for the confirmation that the Resource Group has been successfully deleted. If you don't wait, and the delete fails for some reason, you may be left with resources running that were not expected. You can monitor using the Notifications dialog, which is accessible from the Alarm icon.

    ![The Notifications dialog box has a message stating that the resource group is being deleted.](images/cleanup-delete-resource-group-notification-01.png 'Notifications dialog box')

5. When the Notification indicates success, the cleanup is complete.

    ![The Notifications dialog box has a message stating that the resource group has been deleted.](images/cleanup-delete-resource-group-notification-02.png 'Notifications dialog box')

You should follow all steps provided _after_ attending the Hands-on lab.
