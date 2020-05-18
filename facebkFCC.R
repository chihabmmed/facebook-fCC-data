#NOTE: you will need the internet to download the required datasets
# ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,





##title: "Predicting The Volume of Clicks on Facebook Links With Machine Learning and Neural Networks Models"
##author: "Mohammed Chihab"
##date: "5/14/2020"

#______ SECTION 01 __________________________________________________________________________
#Install and/or load the required libraries
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse")
library(tidyverse)
if(!require(caret)) install.packages("caret")
library(caret)
if(!require(data.table)) install.packages("data.table")
library(data.table)
if(!require(lubridate)) install.packages("lubridate")
library(lubridate)
if(!require(matrixStats)) install.packages("matrixStats")
library(matrixStats)
if(!require(gam)) install.packages("gam")
library(gam)
if(!require(readr)) install.packages("readr")
library(readr)


#______ SECTION 02 __________________________________________________________________________
################################
# Dowload and Read the data set
################################

# Note: this process could take a couple of minutes

# freeCodeCamp-facebook-page-activity dataset:
#datset as of 5/14/2020: https://raw.githubusercontent.com/chihabmmed/facebook-fCC-data/master/freeCodeCamp-facebook-page-activity.csv
#datset as of 5/14/2020: https://files.gitter.im/FreeCodeCamp/DataScience/l4Qf/freeCodeCamp-facebook-page-activity.csv
#datset as of 5/14/2020: https://raw.githubusercontent.com/freeCodeCamp/open-data/master/facebook-fCC-data/data/freeCodeCamp-facebook-page-activity.csv
ds <- tempfile()
download.file("https://raw.githubusercontent.com/chihabmmed/facebook-fCC-data/master/freeCodeCamp-facebook-page-activity.csv", ds)
facebkFCC_main <- read_csv(ds)

#______ SECTION 03 __________________________________________________________________________
######################################################
# Data Wrangling: Cleaning and setting up the data set 
######################################################

# Create a function that will Split the clicks data into three intervals, and convert them into categories named volume
lmh <- function(x) {
  ifelse(x<450,"low",ifelse(x<=1.16e+03,"medium","high"))
}

# Prepare the dataset 
facebkFCC <- facebkFCC_main %>% 
  # use lubridate to wrangle the date variable, keep only the days, convert it into numerical
  mutate(day = as.numeric(wday(mdy(date)))) %>%
  #Factor the type variable, unclass it and convert it into numerical
  mutate(type = as.numeric(unclass(factor(type))))%>%
  #Convert the reach variable into numerical
  mutate(reach = as.numeric(as.character(reach))) %>%
  #Apply the function lmh created above to split it into three levels, name it volume
  mutate(volume = factor(lmh(clicks))) %>%
  #Select only day, type, reach, volume
  select(day, type, reach, volume)

#______ SECTION 04 __________________________________________________________________________
###############################
# Data Exploration - First Part 
###############################

# Dimensions and properties
dim(facebkFCC)

# The structure of the data
str(facebkFCC)

# First three rows in the facebkFCC dataset
head(facebkFCC, 3)

#The summary function provide us with a brief glimps on our data
summary(facebkFCC)

#The number of predictors in the dataset
dim(facebkFCC[,-4])[2]

#Levels in the test set
levels(facebkFCC$volume)

#Proportion of the volume samples in the test set
prop_volume <- data.frame(
  #Proportion of the "high" volume samples in the test set
  prop_high =  mean(facebkFCC$volume == "high"),
  #Proportion of the "medium" volume samples in the test set
  prop_medium = mean(facebkFCC$volume == "medium"),
  #Proportion of the "low" volume samples in the test set
  prop_low = mean(facebkFCC$volume == "low"))
prop_volume

# Plotting the counts per volume (each volume represents an interval of clicks)
ggplot(aes(x = volume), data = facebkFCC, echo = FALSE) + 
  geom_bar(aes(fill = volume), stat = 'count') +
  xlab('Volume') + ylab('Count') + labs(fill = 'volume')


#______ SECTION 05 __________________________________________________________________________
#######################
# Scaling the data set
#######################

#converting the data into a list of a matrix and a vector
facebkFCC_list <- list(x = data.matrix(facebkFCC[,-4]), y = facebkFCC$volume)

#Scaling the matrix of the predictors
x_centered <- sweep(facebkFCC_list$x, 2, colMeans(facebkFCC_list$x))
x_scaled <- sweep(x_centered, 2, colSds(facebkFCC_list$x), FUN = "/")


#______ SECTION 06 __________________________________________________________________________
###############################################################
# Data Exploration - Second Part 
###############################################################

# Performing principal component analysis of the scaled matrix.

# PCA: proportion of variance
pca <- prcomp(x_scaled)
summary(pca)

#PCA: plotting PCs
data.frame(volume = facebkFCC_list$y, pca$x[,1:3]) %>%
  gather(key = "PC", value = "value", -volume) %>%
  ggplot(aes(PC, value, fill = volume, color = volume)) +
  geom_point()

#PCA: PC boxplot
data.frame(volume = facebkFCC_list$y, pca$x[,1:3]) %>%
  gather(key = "PC", value = "value", -volume) %>%
  ggplot(aes(PC, value, fill = volume)) +
  geom_boxplot()


#______ SECTION 07 __________________________________________________________________________
##########################################
# Training and testing data sets creation
##########################################

# Creating a development training set and a validation training set
set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later
validation_index <- createDataPartition(facebkFCC_list$y, times = 1, p = 0.2, list = FALSE)
validation_x <- x_scaled[validation_index,]
validation_y <- facebkFCC_list$y[validation_index]
dev_x <- x_scaled[-validation_index,]
dev_y <- facebkFCC_list$y[-validation_index]


#Splitting the development training set into a training set and a test set
dev_data <- list(x = dev_x, y = dev_y)

set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later
test_index <- createDataPartition(dev_data$y, times = 1, p = 0.2, list = FALSE)
test_x <- dev_data$x[test_index,]
test_y <- dev_data$y[test_index]
train_x <- dev_data$x[-test_index,]
train_y <- dev_data$y[-test_index]

#______ SECTION 08 __________________________________________________________________________
################################################################################
# Training models to predict the volume of clicks based on day, type and reach
################################################################################

##--------------------- lda
train_lda <- train(train_x, train_y,
                   method = "lda")
lda_preds <- predict(train_lda, test_x)
acc_lda <- mean(lda_preds == test_y)
acc_lda


####--------------------- qda
train_qda <- train(train_x, train_y,
                   method = "qda")
qda_preds <- predict(train_qda, test_x)
acc_qda <- mean(qda_preds == test_y)
acc_qda


####--------------------- knn
set.seed(1, sample.kind = "Rounding")
tuning <- data.frame(k = c(3, 5, 7, 9))
train_knn <- train(train_x, train_y,
                   method = "knn", 
                   tuneGrid = tuning)
train_knn$bestTune
knn_preds <- predict(train_knn, test_x)
acc_knn <- mean(knn_preds == test_y)
acc_knn


####--------------------- rf
set.seed(1, sample.kind = "Rounding")
tuning <- data.frame(mtry = c(3, 5, 7, 9))    # can expand to seq(3, 21, 2), same
train_rf <- train(train_x, train_y,
                  method = "rf",
                  tuneGrid = tuning,
                  importance = TRUE)
train_rf$bestTune
rf_preds <- predict(train_rf, test_x)
acc_rf <- mean(rf_preds == test_y)
acc_rf
varImp(train_rf)


####--------------------- nnet
set.seed(1, sample.kind = "Rounding")
trContol <- trainControl(method = "repeatedcv", number = 10, repeats=10)

train_nnet <- train(train_x, train_y,
                    method = "nnet",
                    trControl= trContol,
                    na.action = na.omit,
                    trace = FALSE)
nnet_preds <-predict(train_nnet, test_x)
acc_nnet <- mean(nnet_preds == test_y)
acc_nnet


####--------------------- pcaNNet
set.seed(1, sample.kind = "Rounding")
trContol <- trainControl(method = "repeatedcv", number = 10, repeats=10)

train_pcaNNet <- train(train_x, train_y,
                       method = "pcaNNet",
                       trControl= trContol,
                       na.action = na.omit,
                       trace = FALSE)
pcaNNet_preds <-predict(train_pcaNNet, test_x)
acc_pcaNNet <- mean(pcaNNet_preds == test_y)
acc_pcaNNet

#______ SECTION 09 __________________________________________________________________________
############################################
# Results and Comparisson of Trained Models
############################################

####--------------------- Models Comparison
models <- c("LDA", "QDA", "K nearest neighbors", "Random forest", "nnet", "pcaNNet")
accuracy <- c(acc_lda,
              acc_qda,
              acc_knn,
              acc_rf,
              acc_nnet,
              acc_pcaNNet)
data.frame(Model = models, Accuracy = accuracy) %>% arrange(Accuracy)



#______ SECTION 10 __________________________________________________________________________
###################################################################################
# Predicting the volume of clicks using our model of choice on a simulated dataset.
###################################################################################

####--------------------- validation using pcaNNet
set.seed(1, sample.kind = "Rounding")
trContol <- trainControl(method = "repeatedcv", number = 10, repeats=10)

val_train_pcaNNet <- train(dev_data$x, dev_data$y,
                           method = "pcaNNet",
                           trControl= trContol,
                           na.action = na.omit,
                           trace = FALSE)
val_pcaNNet_preds <-predict(val_train_pcaNNet, validation_x)
val_pcaNNet_acc <- mean(val_pcaNNet_preds == validation_y)
val_pcaNNet_acc

#------------------- END

