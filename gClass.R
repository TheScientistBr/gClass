# General Classification
#
# Delermando Branquinho Filho - delermando@gmail.com
#
# Problem 
# 1 - Build a classificationproblem, using the columns x, y and z, 
# trying to classify the label column.
#
# a)  Segregate a test and training frame. 
# b)  Use a GLM or Logistic Regression model and show theresults. 
# c)  Use other method of your choice to handle the problem 
# d)  Compare and comment the results on the models used from b) and c) 

#### ATTENTION ######
# I have a lot of comments to use on report for non-tecnnical staff 
#### ATTENTION ######

# Step 0: prepare environment to reproducible research
Sys.info()[1:5]
# Results for my machine
# sysname      release      version     nodename      machine 
# "Windows"   ">= 8 x64" "build 9200" "DELERMANDO"     "x86-64"

sessionInfo()
# R version 3.4.0 (2017-04-21)
# Platform: x86_64-w64-mingw32/x64 (64-bit)
# Running under: Windows >= 8 x64 (build 9200)
#
# Matrix products: default

# locale:
# [1] LC_COLLATE=Portuguese_Brazil.1252  LC_CTYPE=Portuguese_Brazil.1252   
# [3] LC_MONETARY=Portuguese_Brazil.1252 LC_NUMERIC=C                      
# [5] LC_TIME=Portuguese_Brazil.1252    
#
# attached base packages:
#        [1] stats     graphics  grDevices utils     datasets  methods   base     
#
# loaded via a namespace (and not attached):
#        [1] compiler_3.4.0 tools_3.4.0  
#
# Loading libraries to work with the most classification algorithms
#
# This package aims to aid practitioners and researchers in using the latest 
# research for analysis of both normally and non-normally distributed return streams.
is.installed <- function(mypkg){
        is.element(mypkg, installed.packages()[,1])
} 

if(!is.installed("PerformanceAnalytics"))
        install.packages("PerformanceAnalytics", dep = TRUE)
library(PerformanceAnalytics)

# The caret package (short for Classification and regression training) 
# is a set of functions that attempt to streamline the process for 
# creating predictive models.
if(!is.installed("caret"))
        install.packages("caret", dep = TRUE)
library("caret")
### Attention ###
# The other 10 algorithm will use other packages, but the train fuction at caret package
# will try to install if the packages are not installed, please, strike "y"- yes
# when you asked for this issues.

# After you asked for split dataset in tranning and test I'll use this:
# We will try to have more control over the computational nuances 
# of the training and test function, both the number of folds and the 
# number of resampling iterations are equal to 10. Only for repeated 
# cross-validation in k-fold: the number of complete sets of folds for 
# compute it 3. This is more efficient than we have tried so far.

control <- trainControl(method="repeatedcv", number=10, repeats=3)

# Defining seed pattern for random number generation for reproducible search
seed <- 7

# Defining the metric for algorithm accuracy
metric <- "Accuracy"

# Now we can start with the research
# Step 1: First let's retrieve dataset
dataset <- read.csv(file = "data/df_points.txt",header = T,stringsAsFactors = F,sep = "\t")

# Determining structure of the dataset
str(dataset)

# Deleting first column because is not belong to dataset and was not
# defined into the requested problem above
# transformation #1
dataset <- dataset[,-1]

# Step 2. Data Exploration
# Data exploration is an approach similar to initial data analysis, 
# whereby a data analyst uses visual exploration to understand what 
# is in a dataset and the characteristics of the data, rather than 
# through traditional data management systems.

# Reading the six firstest lines from dataset
head(dataset)
# We can see numbers negatives and positives named x, y and z, 
# after all, we have a label
# Looking for a range of this variables without label
# Range for variables:
summary(dataset[,1:3])

# Looking for correlation between this variables
# using PerformanceAnalytics package
chart.Correlation(dataset[,1:3])

# Look: there is no correlation between them,
# all variables are uniform distribution

# Step 3. Applying algorithms for classification

# Like you ask we try to a simple method like split dataset
# with 70% for training and 30% to test using glm algorithm
set.seed(seed)
split=0.70
trainIndex <- createDataPartition(dataset$label, p=split, list=FALSE)
data_train <- dataset[ trainIndex,]
data_test <- dataset[-trainIndex,]
# train a glm model
model <- glm(label ~ ., data=data_train)
# make predictions
predictions <- round(predict(model, data_test))
# summarize results
confusionMatrix(predictions, data_test$label)

#### The results ######
# Confusion Matrix and Statistics
# Reference ###########
# Prediction    0    1
# 0  635  458
# 1  865 1042
# 
# Accuracy : 0.559          
# 95% CI : (0.541, 0.5769)
# No Information Rate : 0.5            
# P-Value [Acc > NIR] : 5.513e-11      
# 
# Kappa : 0.118          
# Mcnemar's Test P-Value : < 2.2e-16      
# 
# Sensitivity : 0.4233         
# Specificity : 0.6947         
# Pos Pred Value : 0.5810         
# Neg Pred Value : 0.5464         
# Prevalence : 0.5000         
# Detection Rate : 0.2117         
# Detection Prevalence : 0.3643         
# Balanced Accuracy : 0.5590         
# 
# 'Positive' Class : 0   
#################################################################

# The results are very few 
# We'll try another methods like Repeated k-fold Cross Validation
# as follow ...

###################### trying another methods and algorithms ######################
# In all algorithms below, we set the seed for reproducible research
set.seed(seed)

# transformation #2
# We are going to convert the label variable to factor, 
# it is easier to apply most classification 
# algorithms, because it was loaded as numeric
# after that we have two levels
dataset$label <- as.factor(dataset$label)

# Logistic Regression - as like was asked into problem above
fit.glm <- train(label ~., data=dataset, method="glm", metric=metric, trControl=control)

# Logistic regression 
set.seed(seed)
fit.lda <- train(label ~., data=dataset, method="lda", 
                 metric=metric, preProc=c("center", "scale"), trControl=control)
# GLMNET
set.seed(seed)
fit.glmnet <- train(label ~., data=dataset, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)

# SVM Radial
set.seed(seed)
fit.svmRadial <- train(label ~., data=dataset, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)

# kNN
set.seed(seed)
fit.knn <- train(label ~., data=dataset, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)

# Naive Bayes
set.seed(seed)
fit.nb <- train(label ~., data=dataset, method="nb", metric=metric, trControl=control)

# CART
set.seed(seed)
fit.cart <- train(label ~., data=dataset, method="rpart", metric=metric, trControl=control)

# C5.0
set.seed(seed)
fit.c50 <- train(label ~., data=dataset, method="C5.0", metric=metric, trControl=control)

# Bagged CART
set.seed(seed)
fit.treebag <- train(label ~., data=dataset, method="treebag", metric=metric, trControl=control)

# Random Forest
set.seed(seed)
fit.rf <- train(label ~., data=dataset, method="rf", metric=metric, trControl=control)

# Stochastic Gradient Boosting (Generalized Boosted Modeling)
set.seed(seed)
fit.gbm <- train(label ~., data=dataset, method="gbm", metric=metric, trControl=control, verbose=FALSE)

results <- resamples(list(lda=fit.lda, logistic=fit.glm, glmnet=fit.glmnet,
                          svm=fit.svmRadial, knn=fit.knn, nb=fit.nb, cart=fit.cart, c50=fit.c50,
                          bagging=fit.treebag, rf=fit.rf, gbm=fit.gbm))

# Table comparison
summary(results)

# boxplot comparison
bwplot(results)

# Dot-plot comparison
dotplot(results)

# The comparison of points follows the boxplot including the variation of the minimum and maximum from the average.
# 
# ## Conclusion
# 
# The Random Forest (RF) algorithm for regression and classification has considerably gained popularity since its introduction in 2001. Meanwhile, it has grown to a standard classification approach competing with logistic regression in many innovation-friendly scientific fields.
# 
# In this context, we present a good scale benchmarking experiment based on one datasets comparing the prediction performance of the original version of RF with default parameters and Logistic Rregression as binary classification tools. Most importantly, the design of our benchmark experiment is inspired from clinical trial methodology, thus avoiding common pitfalls and major sources of biases.
# 
# **Random Forest** (RF) performed better than **Logistic Regression** (LR) according to the considered accuracy measured in of the datasets. The mean difference between **RF** and **LR** was $0.22$ percentual points for the accuracy, and $0.443$ of Kappa, all measures thus suggesting a significantly better performance of **RF**. As a side-result of our benchmarking experiment, we observed that the results were noticeably dependent on the inclusion criteria used to select the example datasets (cross-validation against split dataset), thus emphasizing the importance of clear statements regarding this dataset selection process. We also stress that neutral studies similar to ours, based on a high number of datasets to training and test and carefully designed, will be necessary in the future to evaluate further variants, implementations or parameters of random forests which may yield improved accuracy compared to the original version with default values.
# 
# *Sources*
#         Some explanatory texts, such as CART, RF among others were taken from the Internet (Wikipedia).

