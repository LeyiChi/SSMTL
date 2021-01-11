
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
##################################### No.1 Cox survival #####################################
# -------------------------------------------------------------------------------------------

###################################### modeling ##################################
model <- coxph(Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
             data = train, x = TRUE)


####################################### cindex ###################################
concordance(model)
concordance(model, newdata = test)



####################################### ibs ###################################
# Using pec for IBS estimation
ibs.train <- pec(object = model,
                 formula = Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
                 cens.model = "marginal", times = c(12,24,36,48,60,72,84,96), exact = TRUE,
                 data = train, verbose = F, maxtime = 200)
ibs.train.v <- crps(ibs.train)[2]
ibs.train.v


ibs.test <- pec(object = model,
                 formula = Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
                 cens.model = "marginal", times = c(12,24,36,48,60,72,84,96), exact = TRUE,
                 data = test, verbose = F, maxtime = 200)
ibs.test.v <- crps(ibs.test)[2]
ibs.test.v


####################################### bootstrap ###################################
Beta.model <- function(data, indices){
  d <- data[indices,]
  model <- coxph(Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
                 data = d,
                 x = TRUE)
  cindex.train <- concordance(model)$concordance
  
  cindex.test <- concordance(model, newdata = test)$concordance
  
  
  ibs.train <- pec(object = model,
                   formula = Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
                   cens.model = "marginal", times = c(12,24,36,48,60,72,84,96), exact = TRUE,
                   data = d,
                   verbose = F, maxtime = 200)
  ibs.train.v <- crps(ibs.train)[2]
  
  
  ibs.test <- pec(object = model,
                  formula = Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
                  cens.model = "marginal", times = c(12,24,36,48,60,72,84,96), exact = TRUE,
                  data = test, verbose = F, maxtime = 200)
  ibs.test.v <- crps(ibs.test)[2]

  return(c(cindex.train, cindex.test, ibs.train.v, ibs.test.v))
}

result.ci <- boot(data = train, statistic = Beta.model, R = 100)
save(result.ci, file = "../data/result.ci.cox.R")
load(file = "../data/result.ci.cox.R")


round(mean(result.ci$t[, 1]), 4)
round(mean(result.ci$t[, 2]), 4)
round(mean(result.ci$t[, 3]), 4)
round(mean(result.ci$t[, 4]), 4)


get_cis <- function(idx){
  boots.train.cindex.rsf <- result.ci
  boots.train.cindex.rsf$t0 <- result.ci$t0[idx]
  boots.train.cindex.rsf$t <- as.matrix(result.ci$t[, idx])
  res <- boot.ci(boots.train.cindex.rsf, conf = 0.95, type = "perc")
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


