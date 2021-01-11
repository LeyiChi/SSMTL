
# install.packages("BayesFactor")
# install.packages("PMCMR")
setwd("F:\\007-research\\002-papers\\[2019-01 SCI NO 3 深度生存分析]\\revision-03-csd-crc")
rm(list = ls())

library(PMCMR)
library(ggplot2)
options(scipen=200)

set.seed(100)
# ------------------------------------------- model performance ---------------------
load("./data/result.ci.fgr.R")
res.fg <- data.frame("cindex" = result.ci$t[, 2], "ibs" = result.ci$t[, 4])
load("./data/result.ci.rsf.R")
res.rsf <- data.frame("cindex" = result.ci.rsf$t[, 2], "ibs" = result.ci.rsf$t[, 4])
res.smlp <- read.csv("./data/results.ci.smlp.csv")
res.deephit <- read.csv("./data/results.ci.deephit.csv")
res.ssmtlr <- read.csv("./data/results.ci.ssmtlr.csv")



auc_results <- data.frame("fg" = res.fg$cindex, 
                          "rsf" = res.rsf$cindex,
                          "smlp" = res.smlp$cindex,
                          "deephit" = res.deephit$cindex,
                          "ssmtlr" = res.ssmtlr$cindex)

# --------------------------------------------------- Friedman rank sum test -------------------------------------------------

auc_results_matrix <- as.matrix(auc_results)
friedman.test(auc_results_matrix)


# ------------------------------------------------------- Nemenyi test ------------------------------------------------
auc <- posthoc.friedman.nemenyi.test(auc_results_matrix)
summary(auc)





ibs_results <- data.frame("fg" = res.fg$ibs, 
                          "rsf" = res.rsf$ibs, 
                          "smlp" = res.smlp$ibs,
                          "deephit" = res.deephit$ibs,
                          "ssmtlr" = res.ssmtlr$ibs)



# --------------------------------------------------- Friedman rank sum test -------------------------------------------------

ibs_results_matrix <- as.matrix(ibs_results)
friedman.test(ibs_results_matrix)


# ------------------------------------------------------- Nemenyi test ------------------------------------------------
ibs <- posthoc.friedman.nemenyi.test(ibs_results_matrix)

summary(ibs)






