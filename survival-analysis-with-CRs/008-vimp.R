
setwd("~/SSMTL")
rm(list = ls())

library(ggplot2)

temp <- read.csv('../data/007-data_crs_train.csv', stringsAsFactors = TRUE)
data <- read.csv('../data/ssmtlr_vimp.csv')
data <- 0.8364 - data

v.age <- mean(data$vimp[1])
v.plns <- mean(data$vimp[2])
v.race <- mean(data$vimp[3:5])
v.site <- mean(data$vimp[6:8])
v.hist <- mean(data$vimp[9:10])
v.grade <- mean(data$vimp[11:14])
v.ajcc7t <- mean(data$vimp[15:19])
v.ajcc7n <- mean(data$vimp[20:24])
v.ajcc7m <- mean(data$vimp[25:26])
v.surgery <- mean(data$vimp[27:32])
v.radiation <- mean(data$vimp[33:34])


vimps <- c(v.age, v.plns, v.race, v.site, v.hist, v.grade, v.ajcc7t, v.ajcc7n, v.ajcc7m, v.surgery, v.radiation)
vimps <- round(vimps/max(vimps), 2)

vimpdata <- data.frame('variables' = c('Age', "PLNs", "Race", "Site", "Histology", "Grade", "AJCC7T", "AJCC7N", "AJCC7M", "Surgery", "Radiation"),
                       'vimp' = vimps)

tiff(file = "../results/ssmtlr_vimp.tiff", res = 300, width = 800, height = 600, compression = "lzw")
ggplot(vimpdata, aes(x = reorder(variables, vimp, order = TRUE), y = vimp)) +
  geom_bar(stat="identity", fill="lightblue", colour="black", size = 0.3) + coord_flip() +
  geom_text(aes(label = vimps, 
                vjust = 0.4, hjust = -0.3), size = 1.8, show.legend = FALSE) + 
  ggtitle("E. CRC with CRs") +
  labs(x = "Variables", y = "Variable importance") + 
  ylim(0, 1) + 
  theme(plot.title = element_text(color="black", size=7, face="bold", hjust = 0.5)) + 
  theme(axis.text = element_text(size = 6, color="black", vjust=0.5, hjust=0.5)) +
  theme(axis.title = element_text(size = 7, color="black", face= "bold", vjust=0.5, hjust=0.5))
dev.off()

