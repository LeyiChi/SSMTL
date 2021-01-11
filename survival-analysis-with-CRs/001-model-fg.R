
setwd("F:\\007-research\\002-papers\\[2019-01 SCI NO 3 深度生存分析]")

rm(list=ls())

library(survival)
library(foreign)
library(rms)
library(ROCR)
library(DMwR)
library(cluster)
library(riskRegression)
library(caret)
library(parallel)
library(cmprsk)
library(aod)
library(pec)
library(gbm)
library(boot)
options(scipen=200)
options(rf.cores = -1)


# ******************************************** seer *************************************************
#####################################################################################################
########################################### 载入数据 ################################################
#####################################################################################################
# traindata <- read.csv("F:\\Paper\\NN-Survival\\DL4Surv\\revision-02\\data\\data_imputed_train.csv")
# testdata <- read.csv("F:\\Paper\\NN-Survival\\DL4Surv\\revision-02\\data\\data_imputed_test.csv")
load(file = "./revision-03-csd-crc/data/007-data_crs_train.R")
load(file = "./revision-03-csd-crc/data/007-data_crs_test.R")

names(train)
# [1] "race"          "sex"           "age"           "marital"       "site"          "grade"         "ajcc7t"        "ajcc7n"       
# [9] "ajcc7m"        "positivelymph" "time"          "os"            "hist"          "radiation"     "surgery"       "crstatus" 


################################### No.2  Fine-Gray competing risks #################################
# train$crstatus <- factor(train$crstatus, 0:2, labels=c("censor", "death", "odth"))
# test$crstatus <- factor(test$crstatus, 0:2, labels=c("censor", "death", "odth"))
# 
# train.death <- finegray(Surv(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
#                           positivelymph + surgery + radiation,
#                         data = train, etype = "death")
# test.death <- finegray(Surv(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
#                          positivelymph + surgery + radiation,
#                        data = test, etype = "death")
# 
# 
# model.fg <- coxph(Surv(fgstart, fgstop, fgstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
#                     positivelymph + surgery + radiation,
#                   data = train.death, weight = fgwt, x = TRUE)
# concordance(model.fg, ymax = 96)
# # 0.81 0.00139
# 
# concordance(model.fg, newdata = test.death)
# # 0.8036 0.002154
# 
# 
# times <- seq(1, 107, 1)
# train.surv <- t(survfit(model.fg, newdata = train.death)$surv[times, ])
# test.surv <- t(survfit(model.fg, newdata = test.death)$surv[times, ])
# 
# 
# ibs.train.rsf <- pec(object = train.surv,
#                      formula = Hist(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
#                        positivelymph + surgery + radiation,
#                      cens.model = "marginal", cause = "death", exact = TRUE,
#                      data = train.death, verbose = F)



model <- FGR(Hist(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
               positivelymph + surgery + radiation,
             data = train,
             cause = 1)
# pred.fgr <- predict(model, newdata = train, times = seq(1, 107, 1))
cindex.train  <- pec::cindex(model,
                             formula = Hist(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
                               positivelymph + surgery + radiation,
                             data = train,
                             cens.model = "marginal", cause = 1,
                             eval.times = 107)

cindex.train <- cindex.train$AppCindex$FGR
cindex.train

cindex.test  <- pec::cindex(model,
                             formula = Hist(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
                               positivelymph + surgery + radiation,
                             data = test,
                             cens.model = "marginal", cause = 1,
                             eval.times = 107)

cindex.test <- cindex.test$AppCindex$FGR
cindex.test



ibs.train <- pec(object = model,
                     formula = Hist(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
                       positivelymph + surgery + radiation,
                     cens.model = "marginal", cause = 1, exact = TRUE,
                     data = train, verbose = F)
ibs.v.train <- crps(ibs.train)[2]
ibs.v.train

ibs.test <- pec(object = model,
                 formula = Hist(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
                   positivelymph + surgery + radiation,
                 cens.model = "marginal", cause = 1, exact = TRUE,
                 data = test, verbose = F)
ibs.v.test <- crps(ibs.test)[2]
ibs.v.test




####################################### bootstrap ###################################
Beta.model <- function(data, indices){
  d <- data[indices,]
  model <- FGR(Hist(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
                 positivelymph + surgery + radiation,
               data = d,
               cause = 1)
  cindex.train  <- pec::cindex(model,
                               formula = Hist(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
                                 positivelymph + surgery + radiation,
                               data = d,
                               cens.model = "marginal", cause = 1,
                               eval.times = 107)
  
  cindex.train <- cindex.train$AppCindex$FGR
  
  cindex.test  <- pec::cindex(model,
                              formula = Hist(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
                                positivelymph + surgery + radiation,
                              data = test,
                              cens.model = "marginal", cause = 1,
                              eval.times = 107)
  
  cindex.test <- cindex.test$AppCindex$FGR
  
  
  ibs.train <- pec(object = model,
                   formula = Hist(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
                     positivelymph + surgery + radiation,
                   cens.model = "marginal", cause = 1, exact = TRUE,
                   data = d,
                   verbose = F)
  ibs.v.train <- crps(ibs.train)[2]
  
  ibs.test <- pec(object = model,
                  formula = Hist(time, crstatus) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + 
                    positivelymph + surgery + radiation,
                  cens.model = "marginal", cause = 1, exact = TRUE,
                  data = test, 
                  verbose = F)
  ibs.v.test <- crps(ibs.test)[2]
  return(c(cindex.train, cindex.test, ibs.v.train, ibs.v.test))
}

result.ci <- boot(data = train, statistic = Beta.model, R = 100)
save(result.ci, file = "./revision-03-csd-crc/data/result.ci.fgr.R")
load(file = "./revision-03-csd-crc/data/result.ci.fgr.R")


round(mean(result.ci$t[,1]), 4)
# 0.791
round(mean(result.ci$t[,2]), 4)
# 0.7814
round(mean(result.ci$t[,3]), 4)
# 0.1149
round(mean(result.ci$t[,4]), 4)
# 0.1178


get_cis <- function(idx){
  boots.train.cindex.rsf <- result.ci
  boots.train.cindex.rsf$t0 <- result.ci$t0[idx]
  boots.train.cindex.rsf$t <- as.matrix(result.ci$t[, idx])
  res <- boot.ci(boots.train.cindex.rsf, conf = 0.95, type = "perc")
  return(res$percent[4:5])
}
# train cindex
round(get_cis(1), 4)
## 0.7879 0.7949

# test cindex
round(get_cis(2), 4)
## 0.7808 0.7823


# train ibs
round(get_cis(3), 4)
## 0.1132 0.1162

# test ibs
round(get_cis(4), 4)
## 0.1177 0.1180




