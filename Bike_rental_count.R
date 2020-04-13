library(ggplot2)
library(dplyr)
getwd()
setwd("C:/Users/MAYANK/Desktop/ed/project2")
df =read.csv("day.csv")
str(df)
### Finding and Removing Missing Values
missing_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
missing_val$Missing_percent = apply(df,2,function(x){sum(is.na(x))})
names(missing_val)[1] =  "Missing_count"
missing_val$Missing_percent = (missing_val$Missing_percent/nrow(df)) * 100
missing_val = missing_val[order(-missing_val$Missing_percent),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1,3)]
missing_val
### Detecting and removing outlier Removing Outliers
############################################Outlier Analysis#############################################
df2 = df
boxplot(df2)
# ## BoxPlots - Distribution and Outlier Check
numeric_index = sapply(df2,is.numeric) #selecting only numeric

numeric_data = df2[,numeric_index]
cnames = colnames(numeric_data)
cnames
#
# #Remove outliers using boxplot method
#Detect and delete outliers from data
for(i in cnames){
  print(i)
  val = df2[,i][df2[,i] %in% boxplot.stats(df2[,i])$out]
  print(length(val))
  df2 = df2[which(!df2[,i] %in% val),]
}
boxplot(df2)
length(df2$instant)
#* Only 655 data points left After Removal of all possible outliers.
cnames
### Checking the Distribution of data
library(psych)
multi.hist(numeric_data)
#### Scaling Variables casual and registered with minmax scaler
# function to normalize
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
df2$casual = normalize(df2$casual)
df2$registered = normalize(df2$registered)

## Model1
### Linear Regression
#### Assumption
#1. There must be a linear relationship between the dependent variable and the independent variables.
#   Scatterplots can show whether there is a linear or curvilinear relationship.
#2. Multivariate Normality-Multiple regression assumes that the residuals are normally distributed.
#3. No Multicollinearity-Multiple regression assumes that the independent variables are not highly
#   correlated with each other.  This assumption is tested using Variance Inflation Factor (VIF) values.
#4. Homoscedasticity-This assumption states that the variance of error terms are similar across the values
#   of the independent variables.  A plot of standardized residuals versus predicted values can show whether
#   points are equally distributed across all values of the independent variables.

df3 = df2[,3:16]
df3
y = df3[,"cnt"]
X = df3[,1:13]

library(caret)
length(df3$yr)
data1 = cbind(X, y)
data1
lin_reg <- lm(y ~ .-1, data = data1)
summary(lin_reg)
### 1. Identifying whether there is a linear relationship or not
#      between the Target and the dependent variables.
p1 <- ggplot(lin_reg, aes(.fitted, .resid)) + geom_point()
p1 <- p1 + stat_smooth(method="loess") + geom_hline(yintercept=0, col="red", linetype="dashed")
p1 <- p1 + xlab("Predicted") + ylab("Residuals")
p1 <- p1 + ggtitle("Residuals vs. Predicted Values") + theme_bw()

df_plt <- data.frame("fitted" = fitted(lin_reg), "observed" = y)
p2 <- ggplot(df_plt, aes(x=fitted, y=observed)) + geom_point()
p2 <- p2 + stat_smooth(method="loess") + geom_abline(intercept = 1, col="red", linetype="dashed")
p2 <- p2 + xlab("Predicted") + ylab("Observed")
p2 <- p2 + ggtitle("Observed vs. Predicted Values") + theme_bw()
library(gridExtra)

grid.arrange(p2, p1, ncol=2)
##### from the above Graph we can see that there is a linear relationship between variables.
#* they're pretty symmetrically distributed, tending to cluster towards the middle of the plot

### 2. Identifying Multivariate Normality.
#* plotting Residuals with theoritical quantiles.
df_resid <- data.frame(resid = resid(lin_reg))
p <- ggplot(df_resid, aes(sample = resid))
p + stat_qq() + stat_qq_line()
resid =  resid(lin_reg)
resid
#* The good fit indicates that normality is a reasonable approximation.

### 3. Identifying and removing multicolinearity from the data

### function to calculate VIF
VIF <- function(linear.model, no.intercept=FALSE, all.diagnostics=FALSE, plot=FALSE) {
  require(mctest)
  if(no.intercept==FALSE) design.matrix <- model.matrix(linear.model)[,-1]
  if(no.intercept==TRUE) design.matrix <- model.matrix(linear.model)
  if(plot==TRUE) mc.plot(design.matrix,linear.model$model[1])
  if(all.diagnostics==FALSE) output <- imcdiag(design.matrix,linear.model$model[1], method='VIF')$idiags[,1]
  if(all.diagnostics==TRUE) output <- imcdiag(design.matrix,linear.model$model[1])
  output
}

VIF(lin_reg)

#### Since atemp has a High VIF score we will remove it from our data set.
data1 = data1[,!(names(data1) %in% c("atemp"))]
#
#Again Running Regression model

data1
lin_reg <- lm(y ~ .-1, data = data1)
summary(lin_reg)

# removing in significant variables whose P-value lies below 0.05 alpha.
rmv = c("season","yr","mnth" ,"holiday","weathersit")
data1 = data1[,!(names(data1) %in% rmv)]
names(data1)
#
# Rerunning regression
lin_reg <- lm(y ~ .-1, data = data1)
summary(lin_reg)


#######
### 4. Detecting Hetroscedaticity
#* Using goldfeld quandt test
library("lmtest")
gqtest(lin_reg)
##### Since p-value is greater than the alpha = 0.05, we can asume that data is homoscedatic.

###################
### Building a final model after satisfying all the assumption.
library(MLmetrics)

rmse=c()
mape=c()
for (i in 1:100){
  #assign 80% of the data to the training set
  train.index = createDataPartition(data1$y,p=0.80,list = F)
  train <- data1[train.index,]
  test = data1[-train.index,]
  #build model using training data
  lmr =lm(y ~.-1,data=train,trials=100,rules=T)
  #calculate accuracy on test data
  lmrpredict = predict(lmr,test[,1:7])
  rmse[i] <- RMSE(lmrpredict, test[,8])
  mape[i] <- MAPE(lmrpredict, test[,8])
}

for (i in 1:1){
  print("1. explains around 99% variance with RMSE and MAPE")
  print(mean(rmse))
  print(mean(mape))
}
#####
## Model2
### Decision Tree Regressor
library(rpart)
#library(MASS)

data2 = data1
rmse=c()
mape=c()
for (i in 1:100){
  #assign 80% of the data to the training set
  train.index = createDataPartition(data2$y,p=0.80,list = F)
  train <- data2[train.index,]
  test = data2[-train.index,]
  #build model using training data
  dtree =rpart(y ~ .,data=train,method = "anova")
  #calculate accuracy on test data
  lmrpredict = predict(dtree,test[,1:7])
  rmse[i] <- RMSE(lmrpredict, test[,8])
  mape[i] <- MAPE(lmrpredict, test[,8])
}
for (i in 1:1){
  print("Decision tree Regressor, RMSE and MAPE")
  print(mean(rmse))
  print(mean(mape))
}
#####
## Model3
### Random Forest Regressor
library(randomForest)
rng = seq(100, 500, by=20)

rmse=rep(0, times =500 )
mape=rep(0, times =500 )

for (i in rng){
  #assign 80% of the data to the training set
  train.index = createDataPartition(data2$y,p=0.80,list = F)
  train <- data2[train.index,]
  test = data2[-train.index,]
  #build model using training data
  rfst =randomForest(x = train[1:7],y =train$y,ntree = 350, mtry = 5, nodesize = 2)
  #calculate accuracy on test data
  lmrpredict = predict(rfst,test[,1:7])
  rmse[i] <- RMSE(lmrpredict, test[,8])
  mape[i] <- MAPE(lmrpredict, test[,8])
}
#### The Final Random Forest Model
for (i in 1:1){
  print("Random forest Regressor, RMSE and MAPE")
  print(mean(rmse))
  print(mean(mape))
}
## Model 4
### Using Distance based method KNN
# finding optimal no. of Neighbour.
neg=1:13
grid <- expand.grid(k=neg)
# train the model
model <- train(y~., data=data2, metric='RMSE',  method="knn", tuneGrid=grid)
print(model)
# after grid search optimal no. of Neighbour =4
rmse=c()
mape=c()
for (i in 1:100){
  #assign 80% of the data to the training set
  train.index = createDataPartition(data2$y,p=0.80,list = F)
  train <- data2[train.index,]
  test = data2[-train.index,]
  #build model using training data
  dtree =knnreg(y ~ .,data=train,method = "anova")
  #calculate accuracy on test data
  lmrpredict = predict(dtree,test[,1:7])
  rmse[i] <- RMSE(lmrpredict, test[,8])
  mape[i] <- MAPE(lmrpredict, test[,8])
}
#### The Final KNN Model
for (i in 1:1){
  print("Decision tree Regressor, RMSE and MAPE")
  print(mean(rmse))
  print(mean(mape))
}

## Conclusion
#We Derived That on applying The following algorithm we came to know that are linear regression model out performs all the other models in terms of:
#1. Explaining the variance,
#2. along with the minimum mean squared error and
#3. minimum mean absolute percentage error.
#### Hence we can use it to predict The daily bike rental count very effectively.
