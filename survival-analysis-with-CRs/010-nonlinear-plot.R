

setwd("~/SSMTL")

rm(list = ls())
library(ggplot2)
library(RColorBrewer)

nonlinear_age <- read.csv("../results/nonlinear_age.csv")
nonlinear_pln <- read.csv("../results/nonlinear_pln.csv")
nonlinear_race_white <- read.csv("../results/nonlinear_race_white.csv")
nonlinear_race_black <- read.csv("../results/nonlinear_race_black.csv")
nonlinear_race_others <- read.csv("../results/nonlinear_race_others.csv")

nonlinear_site_right_colon <- read.csv("../results/nonlinear_site_right_colon.csv")
nonlinear_site_left_colon <- read.csv("../results/nonlinear_site_left_colon.csv")
nonlinear_site_rectum <- read.csv("../results/nonlinear_site_rectum.csv")

nonlinear_hist_adeno <- read.csv("../results/nonlinear_hist_adeno.csv")
nonlinear_hist_others <- read.csv("../results/nonlinear_hist_others.csv")

nonlinear_grade_1 <- read.csv("../results/nonlinear_grade_1.csv")
nonlinear_grade_2 <- read.csv("../results/nonlinear_grade_2.csv")
nonlinear_grade_3 <- read.csv("../results/nonlinear_grade_3.csv")
nonlinear_grade_4 <- read.csv("../results/nonlinear_grade_4.csv")


nonlinear_Tstage_T1 <- read.csv("../results/nonlinear_Tstage_T1.csv")
nonlinear_Tstage_T2 <- read.csv("../results/nonlinear_Tstage_T2.csv")
nonlinear_Tstage_T3 <- read.csv("../results/nonlinear_Tstage_T3.csv")
nonlinear_Tstage_T4a <- read.csv("../results/nonlinear_Tstage_T4a.csv")
nonlinear_Tstage_T4b <- read.csv("../results/nonlinear_Tstage_T4b.csv")


nonlinear_AJCC7N_1 <- read.csv("../results/nonlinear_AJCC7N_1.csv")
nonlinear_AJCC7N_2 <- read.csv("../results/nonlinear_AJCC7N_2.csv")
nonlinear_AJCC7N_3 <- read.csv("../results/nonlinear_AJCC7N_3.csv")
nonlinear_AJCC7N_4 <- read.csv("../results/nonlinear_AJCC7N_4.csv")
nonlinear_AJCC7N_5 <- read.csv("../results/nonlinear_AJCC7N_5.csv")


nonlinear_AJCC7M_1 <- read.csv("../results/nonlinear_AJCC7M_1.csv")
nonlinear_AJCC7M_2 <- read.csv("../results/nonlinear_AJCC7M_2.csv")


nonlinear_Surgery_1 <- read.csv("../results/nonlinear_Surgery_1.csv")
nonlinear_Surgery_2 <- read.csv("../results/nonlinear_Surgery_2.csv")
nonlinear_Surgery_3 <- read.csv("../results/nonlinear_Surgery_3.csv")
nonlinear_Surgery_4 <- read.csv("../results/nonlinear_Surgery_4.csv")
nonlinear_Surgery_5 <- read.csv("../results/nonlinear_Surgery_5.csv")
nonlinear_Surgery_6 <- read.csv("../results/nonlinear_Surgery_6.csv")

nonlinear_radiation_1 <- read.csv("../results/nonlinear_radiation_1.csv")
nonlinear_radiation_2 <- read.csv("../results/nonlinear_radiation_2.csv")


# ------------------------- age ----------------------------
age <- rep(nonlinear_age$age, 5)
year <- rep(c('1-year', '2-year', '3-year', '4-year', '5-year'), each = 26)
CIF <- c(nonlinear_age$X1.year, nonlinear_age$X2.year, nonlinear_age$X3.year,
         nonlinear_age$X4.year, nonlinear_age$X5.year)
df_age <- data.frame(Age = age, Time = year, CIF = CIF * 100)


tiff(file = "../results/non_age.tiff", res = 600, width = 4800, height = 3600, compression = "lzw")
ggplot(data = df_age, mapping = aes(x = Age, y = CIF, color = Time)) +
  geom_line(size = 1.2, aes(color = Time, linetype = Time)) + 
  scale_y_continuous(limits = c(0, 50)) + 
  labs(x = "Age at diagnosis", y = "CIF") +
  theme(legend.position = c(0.15, 0.8)) + 
  theme(legend.text = element_text(size = rel(1.2), color = "black")) +
  theme(legend.title = element_text(size = rel(1.5), color = "black", face = "bold"), legend.position = "top") +
  theme(legend.background = element_rect(fill="transparent")) +
  theme(axis.text = element_text(size = rel(1.2), color = "black")) +
  theme(axis.title = element_text(size = rel(1.5), color = "black", face = "bold"))
dev.off()


# ------------------------- pln ----------------------------
pln <- rep(nonlinear_pln$age, 5)
year <- rep(c('1-year', '2-year', '3-year', '4-year', '5-year'), each = 26)
CIF <- c(nonlinear_pln$X1.year, nonlinear_pln$X2.year, nonlinear_pln$X3.year,
         nonlinear_pln$X4.year, nonlinear_pln$X5.year)
df_pln <- data.frame(PLN = pln, Time = year, CIF = CIF * 100)


tiff(file = "../results/non_pln.tiff", res = 600, width = 4800, height = 3600, compression = "lzw")
ggplot(data = df_pln, mapping = aes(x = PLN, y = CIF, color = Time)) +
  geom_line(size = 1.2, aes(color = Time, linetype = Time)) + 
  scale_y_continuous(limits = c(0, 100)) + 
  labs(x = "positive lymph node", y = "CIF") +
  theme(legend.position = c(0.15, 0.8)) + 
  theme(legend.text = element_text(size = rel(1.2), color = "black")) +
  theme(legend.title = element_text(size = rel(1.5), color = "black", face = "bold"), legend.position = "top") +
  theme(legend.background = element_rect(fill="transparent")) +
  theme(axis.text = element_text(size = rel(1.2), color = "black")) +
  theme(axis.title = element_text(size = rel(1.5), color = "black", face = "bold"))
dev.off()


# -------------------------------------- boxplot ---------------------------
# ------------------------- race ----------------------------
m <- nrow(nonlinear_race_white)
race <- rep(c('White', 'Black', 'Others'), each = m * 5)
year <- rep(rep(c('1-year', '2-year', '3-year', '4-year', '5-year'), each = m), 3)
CIF <- c(nonlinear_race_white$X1, nonlinear_race_white$X3, nonlinear_race_white$X5, 
         nonlinear_race_white$X7, nonlinear_race_white$X9, 
         nonlinear_race_black$X1, nonlinear_race_black$X3, nonlinear_race_black$X5, 
         nonlinear_race_black$X7, nonlinear_race_black$X9, 
         nonlinear_race_others$X1, nonlinear_race_others$X3, nonlinear_race_others$X5, 
         nonlinear_race_others$X7, nonlinear_race_others$X9)
df_race <- data.frame(Race = race, Year = year, CIF = CIF * 100)
df_race$Race <- factor(df_race$Race,
                         levels = c('White', 'Black', 'Others'),
                         ordered = TRUE)

tiff(file = "../results/non_race.tiff", res = 600, width = 4800, height = 3600, compression = "lzw")
ggplot(data = df_race, mapping = aes(x = Year, y = CIF, color = Race)) +
  geom_boxplot(notch = TRUE, outlier.shape = NA) +
  scale_color_manual(values = c('HotPink','Blue', 'Green')) + 
  labs(x = "Time", y = "CIF", COLOR = "Race", shape = "Race") +
  theme(legend.position = c(0.15, 0.8)) +
  theme(legend.text = element_text(size = rel(1.2), color = "black")) +
  theme(legend.title = element_text(size = rel(1.5), color = "black", face = "bold"), legend.position = "top") +
  theme(legend.background = element_rect(fill="transparent")) +
  theme(axis.text = element_text(size = rel(1.2), color = "black")) +
  theme(axis.title = element_text(size = rel(1.5), color = "black", face = "bold"))
dev.off()


# ------------------------- site ----------------------------
m <- nrow(nonlinear_site_right_colon)
site <- rep(c('Right colon', 'Left colon', 'Rectum'), each = m * 5)
year <- rep(rep(c('1-year', '2-year', '3-year', '4-year', '5-year'), each = m), 3)
CIF <- c(nonlinear_site_right_colon$X1, nonlinear_site_right_colon$X3, nonlinear_site_right_colon$X5, 
         nonlinear_site_right_colon$X7, nonlinear_site_right_colon$X9, 
         nonlinear_site_left_colon$X1, nonlinear_site_left_colon$X3, nonlinear_site_left_colon$X5, 
         nonlinear_site_left_colon$X7, nonlinear_site_left_colon$X9, 
         nonlinear_site_rectum$X1, nonlinear_site_rectum$X3, nonlinear_site_rectum$X5, 
         nonlinear_site_rectum$X7, nonlinear_site_rectum$X9)
df_site <- data.frame(Site = site, Year = year, CIF = CIF * 100)
df_site$Site <- factor(df_site$Site,
                       levels = c('Right colon', 'Left colon', 'Rectum'),
                       ordered = TRUE)

tiff(file = "../results/non_site.tiff", res = 600, width = 4800, height = 3600, compression = "lzw")
ggplot(data = df_site, mapping = aes(x = Year, y = CIF, color = Site)) +
  geom_boxplot(notch = TRUE, outlier.shape = NA) +
  scale_color_manual(values = c('HotPink', 'Purple', 'DeepSkyBlue')) +
  labs(x = "Time", y = "CIF", COLOR = "Site", shape = "Site") +
  theme(legend.position = c(0.15, 0.8)) +
  theme(legend.text = element_text(size = rel(1.2), color = "black")) +
  theme(legend.title = element_text(size = rel(1.5), color = "black", face = "bold"), legend.position = "top") +
  theme(legend.background = element_rect(fill="transparent")) +
  theme(axis.text = element_text(size = rel(1.2), color = "black")) +
  theme(axis.title = element_text(size = rel(1.5), color = "black", face = "bold"))
dev.off()


# ------------------------- hist ----------------------------
m <- nrow(nonlinear_hist_adeno)
hist <- rep(c('Adenocarcinoma', 'Others'), each = m * 5)
year <- rep(rep(c('1-year', '2-year', '3-year', '4-year', '5-year'), each = m), 2)
CIF <- c(nonlinear_hist_adeno$X1, nonlinear_hist_adeno$X3, nonlinear_hist_adeno$X5, 
         nonlinear_hist_adeno$X7, nonlinear_hist_adeno$X9, 
         nonlinear_hist_others$X1, nonlinear_hist_others$X3, nonlinear_hist_others$X5, 
         nonlinear_hist_others$X7, nonlinear_hist_others$X9)
df_hist <- data.frame(Histology = hist, Year = year, CIF = CIF * 100)
df_hist$Histology <- factor(df_hist$Histology,
                            levels = c('Adenocarcinoma', 'Others'),
                            ordered = TRUE)

tiff(file = "../results/non_hist.tiff", res = 600, width = 4800, height = 3600, compression = "lzw")
ggplot(data = df_hist, mapping = aes(x = Year, y = CIF, color = Histology)) +
  geom_boxplot(notch = TRUE, outlier.shape = NA) +
  scale_color_manual(values = c('HotPink','Blue')) +
  labs(x = "Time", y = "CIF", COLOR = "Histology", shape = "Histology") +
  theme(legend.position = c(0.25, 0.8)) +
  theme(legend.text = element_text(size = rel(1.2), color = "black")) +
  theme(legend.title = element_text(size = rel(1.5), color = "black", face = "bold"), legend.position = "top") +
  theme(legend.background = element_rect(fill="transparent")) +
  theme(axis.text = element_text(size = rel(1.2), color = "black")) +
  theme(axis.title = element_text(size = rel(1.5), color = "black", face = "bold"))
dev.off()


# ------------------------- grade ----------------------------
m <- nrow(nonlinear_grade_1)
grade <- rep(c('Grade I', 'Grade II',
             'Grade III', 'Grade IV'), each = m * 5)
year <- rep(rep(c('1-year', '2-year', '3-year', '4-year', '5-year'), each = m), 4)
CIF <- c(nonlinear_grade_1$X1, nonlinear_grade_1$X3, nonlinear_grade_1$X5, 
         nonlinear_grade_1$X7, nonlinear_grade_1$X9, 
         nonlinear_grade_2$X1, nonlinear_grade_2$X3, nonlinear_grade_2$X5, 
         nonlinear_grade_2$X7, nonlinear_grade_2$X9, 
         nonlinear_grade_3$X1, nonlinear_grade_3$X3, nonlinear_grade_3$X5, 
         nonlinear_grade_3$X7, nonlinear_grade_3$X9, 
         nonlinear_grade_4$X1, nonlinear_grade_4$X3, nonlinear_grade_4$X5, 
         nonlinear_grade_4$X7, nonlinear_grade_4$X9)
df_grade <- data.frame(Grade = grade, Year = year, CIF = CIF * 100)
df_grade$Grade <- factor(df_grade$Grade,
                       levels = c('Grade I', 'Grade II',
                                  'Grade III', 'Grade IV'),
                       ordered = TRUE)

tiff(file = "../results/non_grade.tiff", res = 600, width = 4800, height = 3600, compression = "lzw")
ggplot(data = df_grade, mapping = aes(x = Year, y = CIF, color = Grade)) +
  geom_boxplot(notch = TRUE, outlier.shape = NA) +
  scale_color_manual(values = c('HotPink','Blue', 'Green', 'Purple')) + 
  labs(x = "Time", y = "CIF", COLOR = "Grade", shape = "Grade") +
  theme(legend.position = c(0.25, 0.8)) +
  theme(legend.text = element_text(size = rel(1.2), color = "black")) +
  theme(legend.title = element_text(size = rel(1.5), color = "black", face = "bold"), legend.position = "top") +
  theme(legend.background = element_rect(fill="transparent")) +
  theme(axis.text = element_text(size = rel(1.2), color = "black")) +
  theme(axis.title = element_text(size = rel(1.5), color = "black", face = "bold"))
dev.off()


# ------------------------- Tstage ----------------------------
m <- nrow(nonlinear_Tstage_T1)
Tstage <- rep(c('T1', 'T2', 'T3', 'T4a', 'T4b'), each = m * 5)
year <- rep(rep(c('1-year', '2-year', '3-year', '4-year', '5-year'), each = m), 5)
CIF <- c(nonlinear_Tstage_T1$X1, nonlinear_Tstage_T1$X3, nonlinear_Tstage_T1$X5, 
         nonlinear_Tstage_T1$X7, nonlinear_Tstage_T1$X9, 
         nonlinear_Tstage_T2$X1, nonlinear_Tstage_T2$X3, nonlinear_Tstage_T2$X5, 
         nonlinear_Tstage_T2$X7, nonlinear_Tstage_T2$X9, 
         nonlinear_Tstage_T3$X1, nonlinear_Tstage_T3$X3, nonlinear_Tstage_T3$X5, 
         nonlinear_Tstage_T3$X7, nonlinear_Tstage_T3$X9, 
         nonlinear_Tstage_T4a$X1, nonlinear_Tstage_T4a$X3, nonlinear_Tstage_T4a$X5, 
         nonlinear_Tstage_T4a$X7, nonlinear_Tstage_T4a$X9,
         nonlinear_Tstage_T4b$X1, nonlinear_Tstage_T4b$X3, nonlinear_Tstage_T4b$X5, 
         nonlinear_Tstage_T4b$X7, nonlinear_Tstage_T4b$X9)
df_Tstage <- data.frame(AJCC7T = Tstage, Year = year, CIF = CIF * 100)
df_Tstage$AJCC7T <- factor(df_Tstage$AJCC7T,
                       levels = c('T1', 'T2', 'T3', 'T4a', 'T4b'),
                       ordered = TRUE)

tiff(file = "../results/non_Tstage.tiff", res = 600, width = 4800, height = 3600, compression = "lzw")
ggplot(data = df_Tstage, mapping = aes(x = Year, y = CIF, color = AJCC7T)) +
  geom_boxplot(notch = TRUE, outlier.shape = NA) +
  scale_color_manual(values = c('HotPink','Blue', 'Green', 'Purple', 'DeepSkyBlue')) +
  labs(x = "Time", y = "CIF", COLOR = "AJCC7T", shape = "AJCC7T") +
  theme(legend.position = c(0.15, 0.8)) +
  theme(legend.text = element_text(size = rel(1.2), color = "black")) +
  theme(legend.title = element_text(size = rel(1.5), color = "black", face = "bold"), legend.position = "top") +
  theme(legend.background = element_rect(fill="transparent")) +
  theme(axis.text = element_text(size = rel(1.2), color = "black")) +
  theme(axis.title = element_text(size = rel(1.5), color = "black", face = "bold"))
dev.off()


# ------------------------- ajcc7n ----------------------------
m <- nrow(nonlinear_AJCC7N_1)
AJCC7N <- rep(c('N0', 'N1a', 'N1b', 'N2a', 'N2b'), each = m * 5)
year <- rep(rep(c('1-year', '2-year', '3-year', '4-year', '5-year'), each = m), 5)
CIF <- c(nonlinear_AJCC7N_1$X1, nonlinear_AJCC7N_1$X3, nonlinear_AJCC7N_1$X5, 
         nonlinear_AJCC7N_1$X7, nonlinear_AJCC7N_1$X9, 
         nonlinear_AJCC7N_2$X1, nonlinear_AJCC7N_2$X3, nonlinear_AJCC7N_2$X5, 
         nonlinear_AJCC7N_2$X7, nonlinear_AJCC7N_2$X9, 
         nonlinear_AJCC7N_3$X1, nonlinear_AJCC7N_3$X3, nonlinear_AJCC7N_3$X5, 
         nonlinear_AJCC7N_3$X7, nonlinear_AJCC7N_3$X9, 
         nonlinear_AJCC7N_4$X1, nonlinear_AJCC7N_4$X3, nonlinear_AJCC7N_4$X5, 
         nonlinear_AJCC7N_4$X7, nonlinear_AJCC7N_4$X9,
         nonlinear_AJCC7N_5$X1, nonlinear_AJCC7N_5$X3, nonlinear_AJCC7N_5$X5, 
         nonlinear_AJCC7N_5$X7, nonlinear_AJCC7N_5$X9)
df_AJCC7N <- data.frame(AJCC7N = AJCC7N, Year = year, CIF = CIF * 100)
df_AJCC7N$AJCC7N <- factor(df_AJCC7N$AJCC7N,
                           levels = c('N0', 'N1a', 'N1b', 'N2a', 'N2b'),
                           ordered = TRUE)

tiff(file = "../results/non_AJCC7N.tiff", res = 600, width = 4800, height = 3600, compression = "lzw")
ggplot(data = df_AJCC7N, mapping = aes(x = Year, y = CIF, color = AJCC7N)) +
  geom_boxplot(notch = TRUE, outlier.shape = NA) +
  scale_color_manual(values = c('HotPink','Blue', 'Green', 'Purple', 'DeepSkyBlue')) +
  labs(x = "Time", y = "CIF", COLOR = "AJCC7N", shape = "AJCC7N") +
  theme(legend.position = c(0.15, 0.8)) +
  theme(legend.text = element_text(size = rel(1.2), color = "black")) +
  theme(legend.title = element_text(size = rel(1.5), color = "black", face = "bold"), legend.position = "top") +
  theme(legend.background = element_rect(fill="transparent")) +
  theme(axis.text = element_text(size = rel(1.2), color = "black")) +
  theme(axis.title = element_text(size = rel(1.5), color = "black", face = "bold"))
dev.off()


# ------------------------- ajcc7m ----------------------------
m <- nrow(nonlinear_AJCC7M_1)
AJCC7M <- rep(c('M0', 'M1'), each = m * 5)
year <- rep(rep(c('1-year', '2-year', '3-year', '4-year', '5-year'), each = m), 2)
CIF <- c(nonlinear_AJCC7M_1$X1, nonlinear_AJCC7M_1$X3, nonlinear_AJCC7M_1$X5, 
         nonlinear_AJCC7M_1$X7, nonlinear_AJCC7M_1$X9, 
         nonlinear_AJCC7M_2$X1, nonlinear_AJCC7M_2$X3, nonlinear_AJCC7M_2$X5, 
         nonlinear_AJCC7M_2$X7, nonlinear_AJCC7M_2$X9)
df_AJCC7M <- data.frame(AJCC7M = AJCC7M, Year = year, CIF = CIF * 100)
df_AJCC7M$AJCC7M <- factor(df_AJCC7M$AJCC7M,
                           levels = c('M0', 'M1'),
                           ordered = TRUE)

tiff(file = "../results/non_AJCC7M.tiff", res = 600, width = 4800, height = 3600, compression = "lzw")
ggplot(data = df_AJCC7M, mapping = aes(x = Year, y = CIF, color = AJCC7M)) +
  geom_boxplot(notch = TRUE, outlier.shape = NA) +
  scale_color_manual(values = c('HotPink','Blue')) +
  labs(x = "Time", y = "CIF", COLOR = "AJCC7M", shape = "AJCC7M") +
  theme(legend.position = c(0.15, 0.8)) +
  theme(legend.text = element_text(size = rel(1.2), color = "black")) +
  theme(legend.title = element_text(size = rel(1.5), color = "black", face = "bold"), legend.position = "top") +
  theme(legend.background = element_rect(fill="transparent")) +
  theme(axis.text = element_text(size = rel(1.2), color = "black")) +
  theme(axis.title = element_text(size = rel(1.5), color = "black", face = "bold"))
dev.off()


# ------------------------- surgery ----------------------------
m <- nrow(nonlinear_Surgery_1)
Surgery <- rep(c('No surgery', 'Local excision', 'Partial resection', 'Subtotal resection', 'Total resection', 'Others'), each = m * 5)
year <- rep(rep(c('1-year', '2-year', '3-year', '4-year', '5-year'), each = m), 6)
CIF <- c(nonlinear_Surgery_1$X1, nonlinear_Surgery_1$X3, nonlinear_Surgery_1$X5, 
         nonlinear_Surgery_1$X7, nonlinear_Surgery_1$X9, 
         nonlinear_Surgery_2$X1, nonlinear_Surgery_2$X3, nonlinear_Surgery_2$X5, 
         nonlinear_Surgery_2$X7, nonlinear_Surgery_2$X9, 
         nonlinear_Surgery_3$X1, nonlinear_Surgery_3$X3, nonlinear_Surgery_3$X5, 
         nonlinear_Surgery_3$X7, nonlinear_Surgery_3$X9, 
         nonlinear_Surgery_4$X1, nonlinear_Surgery_4$X3, nonlinear_Surgery_4$X5, 
         nonlinear_Surgery_4$X7, nonlinear_Surgery_4$X9,
         nonlinear_Surgery_5$X1, nonlinear_Surgery_5$X3, nonlinear_Surgery_5$X5, 
         nonlinear_Surgery_5$X7, nonlinear_Surgery_5$X9,
         nonlinear_Surgery_6$X1, nonlinear_Surgery_6$X3, nonlinear_Surgery_6$X5, 
         nonlinear_Surgery_6$X7, nonlinear_Surgery_6$X9)
df_Surgery <- data.frame(Surgery = Surgery, Year = year, CIF = CIF * 100)
df_Surgery$Surgery <- factor(df_Surgery$Surgery,
                           levels = c('No surgery', 'Local excision', 'Partial resection', 'Subtotal resection', 'Total resection', 'Others'),
                           ordered = TRUE)

tiff(file = "../results/non_Surgery.tiff", res = 600, width = 4800, height = 3600, compression = "lzw")
ggplot(data = df_Surgery, mapping = aes(x = Year, y = CIF, color = Surgery)) +
  geom_boxplot(notch = TRUE, outlier.shape = NA) +
  scale_color_manual(values = c('HotPink','Blue', 'Green', 'Purple', 'DeepSkyBlue', 'Orange')) +
  labs(x = "Time", y = "CIF", COLOR = "Surgery", shape = "Surgery") +
  theme(legend.position = c(0.15, 0.8)) +
  theme(legend.text = element_text(size = rel(1.2), color = "black")) +
  theme(legend.title = element_text(size = rel(1.5), color = "black", face = "bold"), legend.position = "top") +
  theme(legend.background = element_rect(fill="transparent")) +
  theme(axis.text = element_text(size = rel(1.2), color = "black")) +
  theme(axis.title = element_text(size = rel(1.5), color = "black", face = "bold"))
dev.off()


# ------------------------- Radiation ----------------------------
m <- nrow(nonlinear_radiation_1)
Radiation <- rep(c('No need', 'Need'), each = m * 5)
year <- rep(rep(c('1-year', '2-year', '3-year', '4-year', '5-year'), each = m), 2)
CIF <- c(nonlinear_radiation_1$X1, nonlinear_radiation_1$X3, nonlinear_radiation_1$X5, 
         nonlinear_radiation_1$X7, nonlinear_radiation_1$X9, 
         nonlinear_radiation_2$X1, nonlinear_radiation_2$X3, nonlinear_radiation_2$X5, 
         nonlinear_radiation_2$X7, nonlinear_radiation_2$X9)
df_Radiation <- data.frame(Radiation = Radiation, Year = year, CIF = CIF * 100)
df_Radiation$Radiation <- factor(df_Radiation$Radiation,
                           levels = c('No need', 'Need'),
                           ordered = TRUE)

tiff(file = "../results/non_Radiation.tiff", res = 600, width = 4800, height = 3600, compression = "lzw")
ggplot(data = df_Radiation, mapping = aes(x = Year, y = CIF, color = Radiation)) +
  geom_boxplot(notch = TRUE, outlier.shape = NA) +
  scale_color_manual(values = c('HotPink','Blue')) +
  labs(x = "Time", y = "CIF", COLOR = "Radiation", shape = "Radiation") +
  theme(legend.position = c(0.15, 0.8)) +
  theme(legend.text = element_text(size = rel(1.2), color = "black")) +
  theme(legend.title = element_text(size = rel(1.5), color = "black", face = "bold"), legend.position = "top") +
  theme(legend.background = element_rect(fill="transparent")) +
  theme(axis.text = element_text(size = rel(1.2), color = "black")) +
  theme(axis.title = element_text(size = rel(1.5), color = "black", face = "bold"))
dev.off()


