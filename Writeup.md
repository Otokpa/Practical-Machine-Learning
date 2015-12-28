# Prediction Model on personal data activity
Max. M  
22 Dec 2015  


##Summary

In this project, we will explore personal data activity from devices such as Jawbone Up, Nike FuelBand, and Fitbit. More specificaly data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal  is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.

More information is available from the website here:[goupware](http://groupware.les.inf.puc-rio.br/har) 


Download and load the data:

```r
trainUrl <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
testUrl <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
download.file(trainUrl, 'train.csv', method = 'curl')
download.file(testUrl, 'test.csv', method = 'curl')
trainingData <- read.csv('train.csv', na.strings = c('NA','#DIV/0!', '' ))
testingData <- read.csv('test.csv', na.strings = c('NA','#DIV/0!', '' ))
```

##Exploring the data

```r
str(trainingData)
str(testingData)
```
The training and testing dataset have 160 variables. The training dataset is pretty large with 19622 observations while the testing dataset is small with only 20 observations.There are a lot of columns with a majority of NAs. So l'ets get rid of variables with more than 80% of NAs.


```r
# identify columns with more than 80% of NAs
complete <- colSums(is.na(trainingData))> .8*nrow(trainingData)

# remove columns with more than 80% of NAs
trainingData <- trainingData[, !complete]
testingData <- testingData[, !complete]
# check number of NAs by column
sum(is.na(trainingData))
```

```
## [1] 0
```
We are left with variables without NAs,So this reduced our datasets to 60 variables, but we still have some variables we dont need for our prediction like the different timestamps, usernames, etc. So we will take away the firts 7 columns of our datasets:


```r
trainingData <- trainingData[, -(1:7)]
testingData <- testingData[, -(1:7)]
dim(trainingData)
```

```
## [1] 19622    53
```

```r
dim(testingData)
```

```
## [1] 20 53
```

Here we can see that we are left with 53 variables we can work with to make predictions on the `classe` variable.

##Cross validation parameters

We partition our training set into 30% testing and 70% training using the original testing data as validation.


```r
library(caret)
library(lattice)
library(ggplot2)
set.seed(1234)
intrain <- createDataPartition(trainingData$classe, p = .7, list = F)
training <- trainingData[intrain,]
testing <- trainingData[-intrain,]
dim(training)
```

```
## [1] 13737    53
```

```r
dim(testing)
```

```
## [1] 5885   53
```

Here we have created a training set with 13737 observations and a training set with 5885 observation. Lets check the variatioins of the variables and see if there are some we can drop for poor prediction value.

testing for near zero variation:

```r
library(caret)
nsv <- nearZeroVar(trainingData, saveMetrics = T)
nsv
```

```
##                      freqRatio percentUnique zeroVar   nzv
## roll_belt             1.101904     6.7781062   FALSE FALSE
## pitch_belt            1.036082     9.3772296   FALSE FALSE
## yaw_belt              1.058480     9.9734991   FALSE FALSE
## total_accel_belt      1.063160     0.1477933   FALSE FALSE
## gyros_belt_x          1.058651     0.7134849   FALSE FALSE
## gyros_belt_y          1.144000     0.3516461   FALSE FALSE
## gyros_belt_z          1.066214     0.8612782   FALSE FALSE
## accel_belt_x          1.055412     0.8357966   FALSE FALSE
## accel_belt_y          1.113725     0.7287738   FALSE FALSE
## accel_belt_z          1.078767     1.5237998   FALSE FALSE
## magnet_belt_x         1.090141     1.6664968   FALSE FALSE
## magnet_belt_y         1.099688     1.5187035   FALSE FALSE
## magnet_belt_z         1.006369     2.3290184   FALSE FALSE
## roll_arm             52.338462    13.5256345   FALSE FALSE
## pitch_arm            87.256410    15.7323412   FALSE FALSE
## yaw_arm              33.029126    14.6570176   FALSE FALSE
## total_accel_arm       1.024526     0.3363572   FALSE FALSE
## gyros_arm_x           1.015504     3.2769341   FALSE FALSE
## gyros_arm_y           1.454369     1.9162165   FALSE FALSE
## gyros_arm_z           1.110687     1.2638875   FALSE FALSE
## accel_arm_x           1.017341     3.9598410   FALSE FALSE
## accel_arm_y           1.140187     2.7367241   FALSE FALSE
## accel_arm_z           1.128000     4.0362858   FALSE FALSE
## magnet_arm_x          1.000000     6.8239731   FALSE FALSE
## magnet_arm_y          1.056818     4.4439914   FALSE FALSE
## magnet_arm_z          1.036364     6.4468454   FALSE FALSE
## roll_dumbbell         1.022388    84.2065029   FALSE FALSE
## pitch_dumbbell        2.277372    81.7449801   FALSE FALSE
## yaw_dumbbell          1.132231    83.4828254   FALSE FALSE
## total_accel_dumbbell  1.072634     0.2191418   FALSE FALSE
## gyros_dumbbell_x      1.003268     1.2282132   FALSE FALSE
## gyros_dumbbell_y      1.264957     1.4167771   FALSE FALSE
## gyros_dumbbell_z      1.060100     1.0498420   FALSE FALSE
## accel_dumbbell_x      1.018018     2.1659362   FALSE FALSE
## accel_dumbbell_y      1.053061     2.3748853   FALSE FALSE
## accel_dumbbell_z      1.133333     2.0894914   FALSE FALSE
## magnet_dumbbell_x     1.098266     5.7486495   FALSE FALSE
## magnet_dumbbell_y     1.197740     4.3012945   FALSE FALSE
## magnet_dumbbell_z     1.020833     3.4451126   FALSE FALSE
## roll_forearm         11.589286    11.0895933   FALSE FALSE
## pitch_forearm        65.983051    14.8557741   FALSE FALSE
## yaw_forearm          15.322835    10.1467740   FALSE FALSE
## total_accel_forearm   1.128928     0.3567424   FALSE FALSE
## gyros_forearm_x       1.059273     1.5187035   FALSE FALSE
## gyros_forearm_y       1.036554     3.7763735   FALSE FALSE
## gyros_forearm_z       1.122917     1.5645704   FALSE FALSE
## accel_forearm_x       1.126437     4.0464784   FALSE FALSE
## accel_forearm_y       1.059406     5.1116094   FALSE FALSE
## accel_forearm_z       1.006250     2.9558659   FALSE FALSE
## magnet_forearm_x      1.012346     7.7667924   FALSE FALSE
## magnet_forearm_y      1.246914     9.5403119   FALSE FALSE
## magnet_forearm_z      1.000000     8.5771073   FALSE FALSE
## classe                1.469581     0.0254816   FALSE FALSE
```

From this table we see that we dont have any near  zero variation variables we could drop.

##Exploring types of prediction

Use multicore for speed optimization (this workes on mac only, so skip or use libraries for your system).

```r
# Multicore parallel
library(foreach)
library(iterators)
library(parallel)
library(doMC)
registerDoMC(cores=detectCores())
```


As our outcome is categorical (factor of A, B, C, D, E) it is not appropriate for linear models as it c'ant meet the GLM normality assumption. We will start predicting  with trees.

###predicting with trees

Predicting with trees:

```r
set.seed(1234)
modFit <- train(classe ~ ., data = training, method = 'rpart')
modFit$finalModel
```

```
## n= 13737 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130.5 12563 8667 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -33.05 1111   14 A (0.99 0.013 0 0 0) *
##      5) pitch_forearm>=-33.05 11452 8653 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 436.5 9625 6886 A (0.28 0.18 0.24 0.19 0.11)  
##         20) roll_forearm< 123.5 5965 3517 A (0.41 0.18 0.18 0.17 0.059) *
##         21) roll_forearm>=123.5 3660 2435 C (0.08 0.17 0.33 0.23 0.18) *
##       11) magnet_dumbbell_y>=436.5 1827  904 B (0.033 0.51 0.043 0.23 0.19) *
##    3) roll_belt>=130.5 1174   10 E (0.0085 0 0 0 0.99) *
```

```r
modFit
```

```
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.03549995  0.5211683  0.38183908  0.04385012   0.07081586
##   0.06092971  0.4023505  0.18726056  0.06133138   0.09979318
##   0.11738379  0.3288204  0.06989668  0.04030488   0.06327696
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03549995.
```

```r
pred1 <- predict(modFit, newdata = testing)
confusionMatrix(pred1, data = testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1530   35  105    0    4
##          B  486  379  274    0    0
##          C  493   31  502    0    0
##          D  452  164  348    0    0
##          E  168  145  302    0  467
## 
## Overall Statistics
##                                           
##                Accuracy : 0.489           
##                  95% CI : (0.4762, 0.5019)
##     No Information Rate : 0.5317          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3311          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4890   0.5027   0.3279       NA  0.99151
## Specificity            0.9478   0.8519   0.8797   0.8362  0.88641
## Pos Pred Value         0.9140   0.3327   0.4893       NA  0.43161
## Neg Pred Value         0.6203   0.9210   0.7882       NA  0.99917
## Prevalence             0.5317   0.1281   0.2602   0.0000  0.08003
## Detection Rate         0.2600   0.0644   0.0853   0.0000  0.07935
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638  0.18386
## Balanced Accuracy      0.7184   0.6773   0.6038       NA  0.93896
```

Prediction tree without preprocessing gives poor results. We got an accuracy of 49.85%. lets try to add preprocessing, more specifically centering and scaling the data:

```r
set.seed(1234)
modFit <- train(classe ~ ., data = training, method = 'rpart', 
                preProcess= c('center', 'scale'))
modFit$finalModel
```

```
## n= 13737 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 1.052408 12563 8667 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -1.564666 1111   14 A (0.99 0.013 0 0 0) *
##      5) pitch_forearm>=-1.564666 11452 8653 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 0.6623309 9625 6886 A (0.28 0.18 0.24 0.19 0.11)  
##         20) roll_forearm< 0.8265695 5965 3517 A (0.41 0.18 0.18 0.17 0.059) *
##         21) roll_forearm>=0.8265695 3660 2435 C (0.08 0.17 0.33 0.23 0.18) *
##       11) magnet_dumbbell_y>=0.6623309 1827  904 B (0.033 0.51 0.043 0.23 0.19) *
##    3) roll_belt>=1.052408 1174   10 E (0.0085 0 0 0 0.99) *
```

```r
modFit
```

```
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (52), scaled (52) 
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.03549995  0.5211683  0.38183908  0.04385012   0.07081586
##   0.06092971  0.4023505  0.18726056  0.06133138   0.09979318
##   0.11738379  0.3288204  0.06989668  0.04030488   0.06327696
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03549995.
```

```r
pred1 <- predict(modFit, newdata = testing)
confusionMatrix(pred1, data = testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1530   35  105    0    4
##          B  486  379  274    0    0
##          C  493   31  502    0    0
##          D  452  164  348    0    0
##          E  168  145  302    0  467
## 
## Overall Statistics
##                                           
##                Accuracy : 0.489           
##                  95% CI : (0.4762, 0.5019)
##     No Information Rate : 0.5317          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3311          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4890   0.5027   0.3279       NA  0.99151
## Specificity            0.9478   0.8519   0.8797   0.8362  0.88641
## Pos Pred Value         0.9140   0.3327   0.4893       NA  0.43161
## Neg Pred Value         0.6203   0.9210   0.7882       NA  0.99917
## Prevalence             0.5317   0.1281   0.2602   0.0000  0.08003
## Detection Rate         0.2600   0.0644   0.0853   0.0000  0.07935
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638  0.18386
## Balanced Accuracy      0.7184   0.6773   0.6038       NA  0.93896
```
Preprocessing gives the same result as without preprocessing. Lets try the random forest.

####Random Forest
Using the random forest method with default settings takes to much time! We have to somel to speed it up.

random forest  with K-fold cross validation :

```r
tc_cv <- trainControl(method = 'cv', number = 4)
set.seed(1234)
start.time <- Sys.time()
rf_kfold <- train(classe ~ ., data = training, method = 'rf', 
                trControl = tc_cv)
end.time <- Sys.time()
time.taken <- end.time-start.time
time.taken
```

```
## Time difference of 10.94389 mins
```


```r
pred_rf_kfold <- predict(rf_kfold, newdata = testing)
confusionMatrix(pred_rf_kfold, data = testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    2    0    0    0
##          B    9 1129    1    0    0
##          C    0    8 1017    1    0
##          D    0    0   12  951    1
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9941          
##                  95% CI : (0.9917, 0.9959)
##     No Information Rate : 0.2856          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9925          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9946   0.9912   0.9874   0.9979   0.9991
## Specificity            0.9995   0.9979   0.9981   0.9974   0.9998
## Pos Pred Value         0.9988   0.9912   0.9912   0.9865   0.9991
## Neg Pred Value         0.9979   0.9979   0.9973   0.9996   0.9998
## Prevalence             0.2856   0.1935   0.1750   0.1619   0.1839
## Detection Rate         0.2841   0.1918   0.1728   0.1616   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9971   0.9946   0.9928   0.9976   0.9994
```

Random forest K-fold cross validation gives good results with **99.41%** accuracy.

random forest with Bootstrap :

```r
tc_boot <- trainControl(method="boot", number=4)
set.seed(1234)
start.time <- Sys.time()
rf_boot <- train(classe ~ ., data = training, method = 'rf', 
                trControl = tc_boot)
end.time <- Sys.time()
time.taken <- end.time-start.time
time.taken
```

```
## Time difference of 15.36918 mins
```


```r
rf_boot$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.6%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    2    1    0    1 0.001024066
## B   22 2630    6    0    0 0.010534236
## C    0    9 2382    5    0 0.005843072
## D    0    1   21 2228    2 0.010657194
## E    0    1    3    8 2513 0.004752475
```

```r
rf_boot
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (4 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9890216  0.9861090  0.002103472  0.002655301
##   27    0.9895672  0.9868022  0.001952627  0.002454564
##   52    0.9841950  0.9800053  0.004256414  0.005360415
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
pred_rf_boot <- predict(rf_boot, newdata = testing)
confusionMatrix(pred_rf_boot, data = testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B   10 1128    1    0    0
##          C    0    4 1018    4    0
##          D    0    0    6  957    1
##          E    0    0    2    4 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9946          
##                  95% CI : (0.9923, 0.9963)
##     No Information Rate : 0.2862          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9931          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9941   0.9965   0.9912   0.9917   0.9991
## Specificity            1.0000   0.9977   0.9984   0.9986   0.9988
## Pos Pred Value         1.0000   0.9903   0.9922   0.9927   0.9945
## Neg Pred Value         0.9976   0.9992   0.9981   0.9984   0.9998
## Prevalence             0.2862   0.1924   0.1745   0.1640   0.1830
## Detection Rate         0.2845   0.1917   0.1730   0.1626   0.1828
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9970   0.9971   0.9948   0.9951   0.9989
```
Random forest with Bootstrap has an accuracy of **99.46%**, slightly better than k-fold cross validation.

Random forest with repeated K-fold cross validation :

```r
tc_rkf <- trainControl(method="repeatedcv", number=4, repeats = 2)
set.seed(1234)
start.time <- Sys.time()
rf_rkf <- train(classe ~ ., data = training, method = 'rf', 
                trControl = tc_rkf)
end.time <- Sys.time()
time.taken <- end.time-start.time
time.taken
```

```
## Time difference of 27.681 mins
```


```r
rf_rkf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.63%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    3    0    0    1 0.001024066
## B   23 2629    6    0    0 0.010910459
## C    0   11 2377    8    0 0.007929883
## D    0    1   21 2229    1 0.010213144
## E    0    1    4    7 2513 0.004752475
```

```r
rf_rkf
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold, repeated 2 times) 
## Summary of sample sizes: 10301, 10302, 10304, 10304, 10304, 10303, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9913375  0.9890409  0.002142158  0.002711347
##   27    0.9916285  0.9894099  0.002343148  0.002964312
##   52    0.9880617  0.9848964  0.003854687  0.004878083
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
pred_rf_rkf <- predict(rf_rkf, newdata = testing)
confusionMatrix(pred_rf_rkf, data = testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B   11 1127    1    0    0
##          C    0    5 1017    4    0
##          D    0    1    6  955    2
##          E    0    1    2    3 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9915, 0.9957)
##     No Information Rate : 0.2863          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9935   0.9938   0.9912   0.9927   0.9981
## Specificity            1.0000   0.9975   0.9981   0.9982   0.9988
## Pos Pred Value         1.0000   0.9895   0.9912   0.9907   0.9945
## Neg Pred Value         0.9974   0.9985   0.9981   0.9986   0.9996
## Prevalence             0.2863   0.1927   0.1743   0.1635   0.1832
## Detection Rate         0.2845   0.1915   0.1728   0.1623   0.1828
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9967   0.9957   0.9947   0.9954   0.9984
```
Random forest with repeated K-fold cross validation has an accuracy of **99.39%**. 

Of the 3 random forest methods bootstrap and k-fold cross validation yielded the best results, **99.41%** and **99.46%**. 

##out of sample error estimate
We will use the random forest with bootstrap and k-fold cross validation prediction on the validation dataset. The out of sample error for the K-fold method will not be lower than 1- 0.9941 = **0.0059%**, while the bootstrap method should not be lower than 1-0.9946 = **0.0054%**

## Submission data


```r
# predict on validation dataset
validation_rf_kfold <- predict(rf_kfold, newdata = testingData)
validation_rf_kfold
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
validation_rf_boot <- predict(rf_boot, newdata = testingData)
validation_rf_boot
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Predictions with K-fold: B A B A A E D B A A B C B A E E A B B B
Predictions with bootstrap: B A B A A E D B A A B C B A E E A B B B
These predictions are equal.

Submission function:

```r
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
answers <- validation_rf_kfold
pml_write_files(answers)
```

