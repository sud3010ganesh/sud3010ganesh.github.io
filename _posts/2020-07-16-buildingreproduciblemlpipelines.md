---
title: "Building Reproducing ML Pipelines"
permalink: /2020-07-16-buildingreproduciblemlpipelines/
---

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; One of the common problems encountered while building and deploying machine learning models is the lack of reproducibility in the machine learning workflow especially while transitioning from the development to production environment. 

Reproducibility in a machine learning context can be defined as the ability to replicate a machine learning model such that given the same raw data as input, we get the exact same output.

How often have do we hear a data scientist say "But this model works on my machine?" 

![](/images/repml1.png)<!-- -->

This lack of reproducibility can have significant financial costs. 

In machine learning deployment, it is necessary to note that we don't just deploy the final machine learning algorithm but a complete pipeline that has multiple steps. 

Let's start by looking at the different steps in a ML pipeline before we getting into the topic of reproducibility. The first stage involves gathering data from raw data sources, variables are transformed subsequently and relevant features are selected. We then train and tune the model, choosing the best model hyperparameters based on an error metric. At this point, we have a model ready for deployment and integration with other systems in the business. 

It is imperative to ensure that every single stage of the machine learning pipeline we have outlined above produces identical results given the same inputs.

## Reproducibility during data gathering  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Data gathering stage represents the most difficult challenge in reproducibility. The problems here occur when we cannot reproduce the creation of a training dataset in a subsequent period. For example, if databases are designed such that they are updated and overwritten constantly, the values present at a given time may be overwritten in a subsequent time window posing a challenge from the reproducibility stand point. 

In order to ensure reproducibility in the data gathering phase, one good practise is to save a data snapshot being used to train the model at any given instant. This is pretty helpful when the size of the data is not too big. Alternatively, we can try to ensure the database design does not overwrite historical data and we store archives with time stamps such that a view of the data at any point in time can be obtained. 

Another problem that can arise is due to the inherent randomness in the order of retrieving records by SQL. In order to mitigate for the randomness in retrieving data by SQL, we can sort data by a unique identifier at the data gathering phase. 

## Reproducibility during feature creation 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Lack of reproducibility in the feature creation step can arise due to a multitude of factors. Imputation of missing values with statistical values like mean, median or mode produces different results each time if the underlying training data isn't reproducible. Many parameters extracted in the feature engineering stage depend on the training data and this can be solved by ensuring data reproducibility. 

Categorical feature transformations like one hot encoding create issues in situations when there are categories observed in validation and test sets but not seen during training time. Recording the categories to be encoded prior to model training will ensure this problem is mitigated.

Finally, application of coding best practises like tracking feature generation code under version code and publishing them with timestamped hashed versions can minimize the lack of reproducibility in feature creation to a large extent.

## Reproducibility during model building 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In the model building stage recording the order of features, the applied feature transformations e.g., min-max scaling and the hyperparameters are critical in ensuring reproducibility of models. 

For models that have an inherent element of randomness in the training stage like decision trees or neural networks, setting a seed is necessary to recreate an identical model during a subsequent run. Neural networks are particularly tricky in this respect as we ought to carefully set the seed on several occasions to reproduce the random initializations of multiple parameters that it needs for training. 

If the final model that we planned to deploy is not a standalone model but a stacked one, we also have to record the structure of the ensemble. 

## Reproducibility during model deployment 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To ensure full reproducibility in model deployment stage, the software versions should match exactly. Applications must list all third party libraries, dependencies and versions. 

It is really beneficial to leverage dependency management tools like Pipenv, Poetry or Conda to manage our libraries, dependencies and environments. Packaging the ML application in a container is another best practise that helps in mitigating reproducibility issues while transitioning from model development to deployment phase. When possible research, development and deployment should utilise the same language e.g., Python. Utilizing the same language not only ensures reproducibility but also minimizes the overhead between development and deployment as we don't need to code everything from scratch in a different language during the deployment phase.

Finally, prior to building the machine learning model, it is necessary for the data scientist to have a good understanding of how the model will integrated in production and consumed by other systems in the business eventually. This understanding will ensure that ML pipeline is designed with the integration consideration in mind. A major loss of the benefit that the model should provide arises due to erroneous integration of the model with other business systems. 


>  Factoring in the ideas discussed here while designing our ML pipelines can help improve the reproducibility of ML workflows. 


