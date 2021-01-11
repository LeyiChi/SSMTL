
setwd("~/SSMTL/survival-analysis-without-CRs")
rm(list = ls())

library(PMCMR)
library(ggplot2)
options(scipen=200)

set.seed(100)
# ------------------------------------------- model performance ---------------------
load("../data/result.ci.cox.R")
res.cox <- data.frame("cindex" = result.ci$t[, 2], "ibs" = result.ci$t[, 4])
load("../data/result.ci.aft.R")
res.aft <- data.frame("cindex" = result.ci$t[, 2], "ibs" = result.ci$t[, 4])
load("../data/result.ci.rsf.R")
res.rsf <- data.frame("cindex" = result.ci.rsf$t[, 2], "ibs" = result.ci.rsf$t[, 4])
load("../data/result.ci.gbm.Rdata")
res.gbm <- data.frame("cindex" = result.ci.gbm$t[, 2], "ibs" = result.ci.gbm$t[, 4])
res.deepsurv <- read.csv("../data/results.ci.deepsurv.csv")
res.deephit <- read.csv("../data/results.ci.deephit.csv")
res.mtlr <- read.csv("../data/results.ci.mtlr.csv")
res.loghazard <- read.csv("../data/results.ci.loghazard.csv")
res.pmf <- read.csv("../data/results.ci.pmf.csv")
res.ssmtlr <- read.csv("../data/results.ci.ssmtlr.csv")



auc_results <- data.frame("cox" = res.cox$cindex, 
                          "aft" = res.aft$cindex,
                          "rsf" = res.rsf$cindex,
                          "gbm" = res.gbm$cindex,
                          "deepsurv" = res.deepsurv$cindex,
                          "deephit" = res.deephit$cindex,
                          "mtlr" = res.mtlr$cindex,
                          "loghazard" = res.loghazard$cindex,
                          "pmf" = res.pmf$cindex,
                          "ssmtlr" = res.ssmtlr$cindex)

# --------------------------------------------------- Friedman rank sum test -------------------------------------------------

auc_results_matrix <- as.matrix(auc_results)
friedman.test(auc_results_matrix)


# ------------------------------------------------------- Nemenyi test ------------------------------------------------
auc <- posthoc.friedman.nemenyi.test(auc_results_matrix)
summary(auc)
auc$p.value[9, ]



ibs_results <- data.frame("cox" = res.cox$ibs, 
                          "aft" = res.aft$ibs,
                          "rsf" = res.rsf$ibs,
                          "gbm" = res.gbm$ibs,
                          "deepsurv" = res.deepsurv$ibs,
                          "deephit" = res.deephit$ibs,
                          "mtlr" = res.mtlr$ibs,
                          "loghazard" = res.loghazard$ibs,
                          "pmf" = res.pmf$ibs,
                          "ssmtlr" = res.ssmtlr$ibs)



# --------------------------------------------------- Friedman rank sum test -------------------------------------------------

ibs_results_matrix <- as.matrix(ibs_results)
friedman.test(ibs_results_matrix)


# ------------------------------------------------------- Nemenyi test ------------------------------------------------
ibs <- posthoc.friedman.nemenyi.test(ibs_results_matrix)

summary(ibs)
ibs$p.value[9, ]





