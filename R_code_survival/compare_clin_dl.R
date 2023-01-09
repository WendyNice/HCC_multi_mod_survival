library("survival")
# install.packages("survminer")
library("survminer")

test_clin <- read.csv ("./compare_survival/test_survival_clin.csv")
test_clin_dl <- read.csv ("./compare_survival/test_survival_clin_dl.csv")


test_clin$label_event <- as.numeric(test_clin$label_event)
test_clin$time_event <- as.numeric(test_clin$time_event)

test_clin_dl$label_event <- as.numeric(test_clin_dl$label_event)
test_clin_dl$time_event <- as.numeric(test_clin_dl$time_event)

test_merge <- merge(test_clin, test_clin_dl, by.x='ID', by.y='ID')
test_clin <- test_clin[test_clin$ID %in% test_merge$ID, ]
test_clin_dl <- test_clin_dl[test_clin_dl$ID %in% test_merge$ID, ]
# Cox proportional hazards regression 
Surv(test_clin$time_event, test_clin$label_event)
cox_single <- coxph(Surv(time_event, label_event) ~ BCLC+Albumin+AST, data = test_clin)    
cox_muti <- coxph(Surv(time_event, label_event) ~ BCLC+Albumin+AST+dl_ft, data = test_clin_dl)
summary(cox_single)
summary(cox_muti)


# log partial likelihood.
anova(cox_single,cox_muti)


