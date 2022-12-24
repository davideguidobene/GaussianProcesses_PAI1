# PAI_Pr1_GaussianProcesses
Project 1 for the course of Probabilistic Artificial Intelligence. Used Gaussian Process Regression to predict and audit the concentration of fine particulate matter (PM2.5) per cubic meter of air.

# HOW TO EXECUTE
1. Download Docker
2. To run from Linux
  $ bash runner.sh

# SHORT REPORT
The features X are standardized using StandardScaler from SKLearn.
The model used for inference is the Gaussian Proccess, implemented through GaussianProcessRegressor from SKLearn and DotProduct as its kernel.
Notice that the label Y (pollution) is authomatically standardized by GaussianProcessRegressor during training (and the the transformation is reversed during inference), since normalize_Y is set True. Therefore, the use of a StandardScaler is not necessary in this instance.

