
setwd("~/SSMTL/survival-analysis-without-CRs")

rm(list=ls())

library(survival)
library(foreign)
library(rms)
library(ROCR)
library(caret)
library(cluster)
library(randomForestSRC)
library(parallel)
library(cmprsk)
library(aod)
library(pec)
library(gbm)
library(ranger)
library(boot)
library(scoring)
source("./000-BrierScore.R")
source("./000-aft_funcs.R")
options(scipen=200)
options(contrasts=c("contr.treatment", "contr.treatment"))


# ******************************************** seer *************************************************
#####################################################################################################
########################################### load data ###############################################
#####################################################################################################

load(file = "../data/007-data_os_train.R")
load(file = "../data/007-data_os_test.R")


train$os <- ifelse(train$os == 4, 1, 0)
test$os <- ifelse(test$os == 4, 1, 0)



# -------------------------------------------------------------------------------------------
######################################## No.3 AFT model #####################################
# -------------------------------------------------------------------------------------------
###################################### modeling ##################################
model <- survreg(Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
                     dist = 'weibull', data = train)

####################################### cindex ###################################
concordance(model)
concordance(model, newdata = test)


aftMod.train <- AFT(train, train, "weibull")
aftBrierInt.train <- BrierScore(aftMod.train, type = "Integrated", numPoints = 8, integratedBrierTimes = c(12,96))


aftMod.test <- AFT(train, test, "weibull")
aftBrierInt.test <- BrierScore(aftMod.test, type = "Integrated", numPoints = 8, integratedBrierTimes = c(12,96))


####################################### bootstrap ###################################
Beta.model <- function(data, indices){
  d <- data[indices,]
  model <- survreg(Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
                      dist = 'weibull',
                      data = d)
  
  cindex.train <- concordance(model)$concordance
  cindex.test <- concordance(model, newdata = test)$concordance

  aftMod.train <- AFT(d, d, "weibull")
  ibs.train.v <- BrierScore(aftMod.train, type = "Integrated", numPoints = 8, integratedBrierTimes = c(12,96))
  aftMod.test <- AFT(d, test, "weibull")
  ibs.test.v <- BrierScore(aftMod.test, type = "Integrated", numPoints = 8, integratedBrierTimes = c(12,96))
  
  return(c(cindex.train, cindex.test, ibs.train.v, ibs.test.v))
}

result.ci <- boot(data = train, statistic = Beta.model, R = 100)
save(result.ci, file = "../data/result.ci.aft.R")
load(file = "../data/result.ci.aft.R")


round(mean(result.ci$t[, 1]), 4)
round(mean(result.ci$t[, 2]), 4)
round(mean(result.ci$t[, 3]), 4)
round(mean(result.ci$t[, 4]), 4)


get_cis <- function(idx){
  boots.train.cindex <- result.ci
  boots.train.cindex$t0 <- result.ci$t0[idx]
  boots.train.cindex$t <- as.matrix(result.ci$t[, idx])
  res <- boot.ci(boots.train.cindex, conf = 0.95, type = "perc")
  return(res$percent[4:5])
}
# train cindex
round(get_cis(1), 4)

# test cindex
round(get_cis(2), 4)


# train ibs
round(get_cis(3), 4)

# test ibs
round(get_cis(4), 4)

