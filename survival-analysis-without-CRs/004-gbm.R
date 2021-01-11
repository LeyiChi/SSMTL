
setwd("~/SSMTL/survival-analysis-without-CRs")

rm(list=ls())

library(survival)
library(foreign)
library(rms)
library(ROCR)
library(caret)
library(cluster)
library(parallel)
library(cmprsk)
library(aod)
library(pec)
library(gbm)
library(ranger)
library(boot)
library(scoring)
options(scipen=200)
options(rf.cores = -1)

options(contrasts=c("contr.treatment", "contr.treatment"))



# ******************************************** seer *************************************************
#####################################################################################################
########################################### load data ###############################################
#####################################################################################################
load(file = "../data/007-data_os_train.R")
load(file = "../data/007-data_os_test.R")


train$os <- ifelse(train$os == 4, 1, 0)
test$os <- ifelse(test$os == 4, 1, 0)


folds <- createFolds(train$os, k = 5)
gbm_grid <- expand.grid(interaction.depth = c(5, 6, 7, 8, 9, 10), 
                        shrinkage = c(0.1, 0.01, 0.001), 
                        n.trees = c(100,200,300,400,500),
                        n.minobsinnode = c(20, 30, 40, 50, 60),
                        bag.fraction = c(0.6, 0.7, 0.8, 0.9))

### -------------------------------- GBM model -------------------------------------
get_cindex <- function(time, os, riskscore){
  concordanceInfo = survConcordance(Surv(time, os) ~ riskscore)
  concordantPairs= concordanceInfo$stats[1] 
  discordantPairs = concordanceInfo$stats[2] 
  riskTies = concordanceInfo$stats[3]
  timeTies = concordanceInfo$stats[4]
  
  cindex = (concordantPairs+riskTies/2)/(concordantPairs + discordantPairs + riskTies)
  
  return(cindex)
}


# Function for CV
gbm.cv <- function(data, n.trees, interaction.depth, n.minobsinnode, shrinkage, bag.fraction, IBS = FALSE){
  aucs_gbm <- c() # Initiate
  ibss <- c()
  
  for (ifold in 1:length(folds)){
    itr <- unlist(folds[-ifold])
    Xtrain <- data[itr,]
    Xvalid <- data[-itr,]
    # Fit gbm
    model.gbm <- gbm(Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
                    n.cores = -1, distribution = "coxph",
                    data = Xtrain, 
                    n.trees = n.trees, 
                    interaction.depth = interaction.depth,
                    n.minobsinnode = n.minobsinnode,
                    shrinkage = shrinkage,
                    bag.fraction = bag.fraction)
    
    pred.test <- predict(model.gbm, Xvalid, se.fit = FALSE, type = "response", n.trees = n.trees)
    riskscore.test <- exp(pred.test)
    

    auc <- get_cindex(Xvalid$time, Xvalid$os, riskscore.test)
    aucs_gbm <- c(aucs_gbm, auc)
    
  }
  # Return the cindex vector
  if(IBS){
    return(list(aucs_gbm, ibss))
  }
  else{
    return(aucs_gbm)
  }
}

# Loop over the rf_grid to screen best hyperparameters
gbm.cv.results <- data.frame(gbm_grid)
gbm.cv.results$auc <- 0



for(ind in 451:dim(gbm_grid)[1]){
  interaction.depth <- gbm_grid[ind,1]
  shrinkage <- gbm_grid[ind,2]
  n.trees <- gbm_grid[ind,3]
  n.minobsinnode <- gbm_grid[ind,4]
  bag.fraction <- gbm_grid[ind,5]
  
  # Run fivefold CV
  gbm.rets <- gbm.cv(data = train, n.trees = n.trees, interaction.depth = interaction.depth, n.minobsinnode = n.minobsinnode, 
                     shrinkage = shrinkage, bag.fraction = bag.fraction, IBS = FALSE)
  
  gbm.cv.results[ind, 6] <- mean(gbm.rets[[1]])
  
  # Write report
  cat("Model :: ", ind, "/", dim(gbm_grid)[1], "n.trees = ", n.trees, "interaction.depth = ", interaction.depth, 
      "n.minobsinnode = ", n.minobsinnode,
      "shrinkage = ", shrinkage,
      "bag.fraction = ", bag.fraction,
      ":: auc = ", gbm.cv.results[ind, 6], "\n")
  save(gbm.cv.results, file="../data/cv.results.gbm.Rdata")
}

save(gbm.cv.results, file="../data/cv.results.gbm.Rdata")
load(file="../data/cv.results.gbm.Rdata")




ind.best <- which.max(gbm.cv.results$auc)



# -------------------------------------------------------------------------------------------
############################################# No.1 GBM ######################################
# -------------------------------------------------------------------------------------------

###################################### modeling ##################################



model <- gbm(Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
             n.cores = -1, distribution = "coxph",
             data = train, 
             interaction.depth = gbm_grid[ind,1],
             shrinkage = gbm_grid[ind,2],
             n.trees = gbm_grid[ind,3],
             n.minobsinnode = gbm_grid[ind,4],
             bag.fraction = gbm_grid[ind,5])

save(model, file = "../data/model_gbm.Rdata")
load(file = "../data/model_gbm.Rdata")
summary(model)



####################################### cindex ###################################
pred.train <- predict(model, train, se.fit = FALSE, type = "response", n.trees = gbm_grid[ind,1])
riskscore.train <- exp(pred.train)
pred.test <- predict(model, test, se.fit = FALSE, type = "response", n.trees = gbm_grid[ind,1])
riskscore.test <- exp(pred.test)



get_cindex(train$time, train$os, riskscore.train)
get_cindex(test$time, test$os, riskscore.test)



####################################### ibs ###################################
# Using pec for IBS estimation

get_ibs <- function(data, pred){
  lambda0 <- basehaz.gbm(t = data$time, delta = data$os,  t.eval =   sort(unique(data$time)), cumulative = FALSE, f.x = pred , smooth = T)
  hazard <- matrix(lambda0, ncol = 1) %*% matrix(exp(pred), nrow = 1) 
  Hazard <- t(as.data.frame(apply(hazard,2,cumsum)))
  surv <- cbind(rep(1, dim(Hazard)[1]), exp(-1 * Hazard))
  
  
  PredError <- pec(object = surv, # matrix(hazard, ncol = 107),
                   formula = Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
                   cens.model = "marginal", exact = TRUE,
                   data= data, verbose=F)
  
  # print(PredError, times = c(12,24,36,48,60,72,84,96))
  ibs.v <- crps(PredError, times = 96)
  return(ibs.v)
  
}
get_ibs(train, pred.train)
get_ibs(test, pred.test)





####################################### bootstrap ###################################
Beta.gbm <- function(data, indices){
  d <- data[indices,]
  model.gbm <- gbm(Surv(time, os) ~ race + age + site + hist + grade + ajcc7t + ajcc7n + ajcc7m + positivelymph + surgery + radiation,
               n.cores = -1, distribution = "coxph",
               data = d, 
               interaction.depth = gbm_grid[ind,1],
               shrinkage = gbm_grid[ind,2],
               n.trees = gbm_grid[ind,3],
               n.minobsinnode = gbm_grid[ind,4],
               bag.fraction = gbm_grid[ind,5])
  
  pred.train <- predict(model, d, se.fit = FALSE, type = "response", n.trees = gbm_grid[ind,1])
  riskscore.train <- exp(pred.train)
  pred.test <- predict(model.gbm, test, se.fit = FALSE, type = "response", n.trees = gbm_grid[ind,1])
  riskscore.test <- exp(pred.test)
  
  cindex.train <- get_cindex(d$time, d$os, riskscore.train)
  cindex.test <- get_cindex(test$time, test$os, riskscore.test)
  ibs.train.v <- get_ibs(d, pred.train)
  ibs.test.v <- get_ibs(test, pred.test)
  
  return(c(cindex.train, cindex.test, ibs.train.v, ibs.test.v))
}

result.ci.gbm <- boot(data = train, statistic = Beta.gbm, R = 100)
save(result.ci.gbm, file = "../data/result.ci.gbm.Rdata")
load(file = "../data/result.ci.gbm.Rdata")


round(mean(result.ci.gbm$t[, 1]), 4)
round(mean(result.ci.gbm$t[, 2]), 4)
round(mean(result.ci.gbm$t[, 3]), 4)
round(mean(result.ci.gbm$t[, 4]), 4)


get_cis <- function(idx){
  boots.train.cindex.rsf <- result.ci.gbm
  boots.train.cindex.rsf$t0 <- result.ci.gbm$t0[idx]
  boots.train.cindex.rsf$t <- as.matrix(result.ci.gbm$t[, idx])
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

