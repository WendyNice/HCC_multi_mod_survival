library("survival")
# install.packages("survminer")
library("survminer")
# 
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install(version = "3.12")
# library("BiocManager")

# BiocManager::install("survcomp")
# install.packages("nomogramFormula")
library(nomogramFormula)
library(survcomp)
library(rms)
library(Hmisc)
library(openxlsx)


####################################################################
# OS Train
train_df <- read.csv ("D:/Data_analysis/Survival_analysis/SYSU_Cancer_HCC/result_sorting/result/compare/train_df_rfs.csv")
test_df <- read.csv ("D:/Data_analysis/Survival_analysis/SYSU_Cancer_HCC/result_sorting/result/compare/test_df_rfs.csv")
val_df <- read.csv ("D:/Data_analysis/Survival_analysis/SYSU_Cancer_HCC/result_sorting/result/compare/valid_df_rfs.csv")

train_df$label_event <- as.numeric(train_df$label_event)
train_df$time_event <- as.numeric(train_df$time_event)

test_df$label_event <- as.numeric(test_df$label_event)
test_df$time_event <- as.numeric(test_df$time_event)

val_df$label_event <- as.numeric(val_df$label_event)
val_df$time_event <- as.numeric(val_df$time_event)

train_df$time_event <- as.integer((train_df$time_event) / 30)
test_df$time_event <- as.integer((test_df$time_event) / 30)
val_df$time_event <- as.integer((val_df$time_event) / 30)

Traintime<-train_df$time_event;Trainevent<-train_df$label_event
Testtime<-test_df$time_event;Testevent<-test_df$label_event
Exttime<-val_df$time_event;Extevent<-val_df$label_event

TrainSurv<-Surv(Traintime,Trainevent)
TestSurv<-Surv(Testtime,Testevent)
ExtSurv<-Surv(Exttime,Extevent)

# Cox proportional hazards regression 
cox_single <- coxph(Surv(time_event, label_event) ~ BCLC+gender+Albumin, data = train_df)    
cox_muti <- coxph(Surv(time_event, label_event) ~ BCLC+gender+Albumin+dl_ft, data = train_df)  
cox_dl <- coxph(Surv(time_event, label_event) ~ dl_ft, data = train_df)  

dd <- datadist(train_df)
options(datadist = 'dd')
f_single <- cph(Surv(time_event, label_event) ~ BCLC+gender+Albumin, data = train_df, x=T, y=T, surv=T)#多变量

dd <- datadist(train_df)
options(datadist = 'dd')
f_multi <- cph(Surv(time_event, label_event) ~ BCLC+gender+Albumin+dl_ft, data = train_df, x=T, y=T, surv=T)#多变量

dd <- datadist(train_df)
options(datadist = 'dd')
f_dl <- cph(Surv(time_event, label_event) ~ dl_ft, data = train_df, x=T, y=T, surv=T)#多变量


summary(cox_single)
summary(cox_muti)
summary(cox_dl)


# single
CIN_train <- survConcordance(TrainSurv~predict(cox_single,train_df))
CIN_test <- survConcordance(TestSurv~predict(cox_single,test_df))
CIN_ext <- survConcordance(ExtSurv~predict(cox_single,val_df))

# multi
CIN_train2 <- survConcordance(TrainSurv~predict(cox_muti,train_df))
CIN_test2 <- survConcordance(TestSurv~predict(cox_muti,test_df))
CIN_ext2 <- survConcordance(ExtSurv~predict(cox_muti,val_df))

# dl
CIN_train3 <- survConcordance(TrainSurv~predict(cox_dl,train_df))
CIN_test3 <- survConcordance(TestSurv~predict(cox_dl,test_df))
CIN_ext3 <- survConcordance(ExtSurv~predict(cox_dl,val_df))

# p VALUE
# TRAIN
train_pre=list()
train_pre$x=predict(cox_single, train_df)
Ctrain=list(c.index=CIN_train$concordance,n=CIN_train$n,se=CIN_train$std.err,data=list())
Ctrain$data=train_pre

train_pre2=list()
train_pre2$x=predict(cox_muti, train_df)
Ctrain2=list(c.index=CIN_train2$concordance,n=CIN_train2$n,se=CIN_train2$std.err,data=list())
Ctrain2$data=train_pre2

train_pre3=list()
train_pre3$x=predict(cox_dl, train_df)
Ctrain3=list(c.index=CIN_train3$concordance,n=CIN_train3$n,se=CIN_train3$std.err,data=list())
Ctrain3$data=train_pre3

# cindex.comp should put the better one in the first parameter
p_train<-cindex.comp(Ctrain2, Ctrain)
print('p_train')
p_train

p_train2<-cindex.comp(Ctrain3, Ctrain)
print('p_train2')
p_train2

# Test
train_pre=list()
train_pre$x=predict(cox_single, test_df)
Ctrain=list(c.index=CIN_test$concordance,n=CIN_test$n,se=CIN_test$std.err,data=list())
Ctrain$data=train_pre

train_pre2=list()
train_pre2$x=predict(cox_muti, test_df)
Ctrain2=list(c.index=CIN_test2$concordance,n=CIN_test2$n,se=CIN_test2$std.err,data=list())
Ctrain2$data=train_pre2

train_pre3=list()
train_pre3$x=predict(cox_dl, test_df)
Ctrain3=list(c.index=CIN_test3$concordance,n=CIN_test3$n,se=CIN_test3$std.err,data=list())
Ctrain3$data=train_pre3

# cindex.comp should put the better one in the first parameter
p_test<-cindex.comp(Ctrain2, Ctrain)
print('p_test')
p_test

p_test2<-cindex.comp(Ctrain3, Ctrain)
print('p_test2')
p_test2

# valid
train_pre=list()
train_pre$x=predict(cox_single, val_df)
Ctrain=list(c.index=CIN_ext$concordance,n=CIN_ext$n,se=CIN_ext$std.err,data=list())
Ctrain$data=train_pre

train_pre2=list()
train_pre2$x=predict(cox_muti, val_df)
Ctrain2=list(c.index=CIN_ext2$concordance,n=CIN_ext2$n,se=CIN_ext2$std.err,data=list())
Ctrain2$data=train_pre2

train_pre3=list()
train_pre3$x=predict(cox_dl, val_df)
Ctrain3=list(c.index=CIN_ext3$concordance,n=CIN_ext3$n,se=CIN_ext3$std.err,data=list())
Ctrain3$data=train_pre3


# cindex.comp should put the better one in the first parameter
p_val<-cindex.comp(Ctrain2, Ctrain)
print('p_val')
p_val

p_val2<-cindex.comp(Ctrain3, Ctrain)
print('p_val2')
p_val2



# single
print('single')
CIN_train <- survConcordance(TrainSurv~predict(cox_single,train_df))
CIN_test <- survConcordance(TestSurv~predict(cox_single,test_df))
CIN_ext <- survConcordance(ExtSurv~predict(cox_single,val_df))

cindex_train <- CIN_train$concordance
cindex_train$low<-CIN_train$concordance-1.96*CIN_train$std.err
cindex_train$high<-CIN_train$concordance+1.96*CIN_train$std.err
cindex_test <- CIN_test$concordance
cindex_test$low<-CIN_test$concordance-1.96*CIN_test$std.err
cindex_test$high<-CIN_test$concordance+1.96*CIN_test$std.err
cindex_ext <- CIN_ext$concordance
cindex_ext$low<-CIN_ext$concordance-1.96*CIN_ext$std.err
cindex_ext$high<-CIN_ext$concordance+1.96*CIN_ext$std.err
cindex_train
cindex_test
cindex_ext

# multi
print('multi')
CIN_train2 <- survConcordance(TrainSurv~predict(cox_muti,train_df))
CIN_test2 <- survConcordance(TestSurv~predict(cox_muti,test_df))
CIN_ext2 <- survConcordance(ExtSurv~predict(cox_muti,val_df))

cindex_train2 <- CIN_train2$concordance
cindex_train2$low<-CIN_train2$concordance-1.96*CIN_train2$std.err
cindex_train2$high<-CIN_train2$concordance+1.96*CIN_train2$std.err
cindex_test2 <- CIN_test2$concordance
cindex_test2$low<-CIN_test2$concordance-1.96*CIN_test2$std.err
cindex_test2$high<-CIN_test2$concordance+1.96*CIN_test2$std.err
cindex_ext2 <- CIN_ext2$concordance
cindex_ext2$low<-CIN_ext2$concordance-1.96*CIN_ext2$std.err
cindex_ext2$high<-CIN_ext2$concordance+1.96*CIN_ext2$std.err
cindex_train2
cindex_test2
cindex_ext2

# dl
print('dl')
CIN_train3 <- survConcordance(TrainSurv~predict(cox_dl,train_df))
CIN_test3 <- survConcordance(TestSurv~predict(cox_dl,test_df))
CIN_ext3 <- survConcordance(ExtSurv~predict(cox_dl,val_df))

cindex_train3 <- CIN_train3$concordance
cindex_train3$low<-CIN_train3$concordance-1.96*CIN_train3$std.err
cindex_train3$high<-CIN_train3$concordance+1.96*CIN_train3$std.err
cindex_test3 <- CIN_test3$concordance
cindex_test3$low<-CIN_test3$concordance-1.96*CIN_test3$std.err
cindex_test3$high<-CIN_test3$concordance+1.96*CIN_test3$std.err
cindex_ext3 <- CIN_ext3$concordance
cindex_ext3$low<-CIN_ext3$concordance-1.96*CIN_ext3$std.err
cindex_ext3$high<-CIN_ext3$concordance+1.96*CIN_ext3$std.err
cindex_train3
cindex_test3
cindex_ext3


# clinic
#nomogram

survival = Survival(f_single)
survival_5 = function(x) survival(24, x)
nom <- nomogram(f_single,fun=list(survival_5),
                funlabel = c("5 year survival"),
                fun.at = c(0.05,seq(0.1,0.9,by=0.05),0.95), lp=FALSE)
plot(nom)
summary(f_single)
#risk score
results <- formula_rd(nomogram = nom)
test_df$points <- points_cal(formula = results$formula,rd=test_df)
train_df$points <- points_cal(formula = results$formula,rd=train_df)
val_df$points <- points_cal(formula = results$formula,rd=val_df)

res.cut <- surv_cutpoint(train_df, time="time_event",event="label_event",variables =c("points"))
cutpoint<-res.cut$cutpoint[1,1]

#Train KM
train_df$CPoints<-train_df$points
train_df$CPoints[train_df$CPoints <=cutpoint] <- 0
train_df$CPoints[train_df$CPoints >cutpoint] <- 1
# write.xlsx(train_df, 'D:\\Data analysis\\Survival_analysis\\SYSU_Cancer_HCC\\result_sorting\\risk_layered\\clinical_train.xlsx')

fml_KM1<-survfit(TrainSurv~CPoints,data=train_df)
x11()
ggsurvplot(fml_KM1,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability (percentage)",xlab = " Time (Months)",
           legend.title="Training cohort")

# TEST KM
test_df$CPoints<-test_df$points
test_df$CPoints[test_df$CPoints <cutpoint] <- 0
test_df$CPoints[test_df$CPoints >=cutpoint] <- 1
# write.xlsx(test_df, 'D:\\Data analysis\\Survival_analysis\\SYSU_Cancer_HCC\\result_sorting\\risk_layered\\clinical_test.xlsx')

fml_KM<-survfit(TestSurv~CPoints,data=test_df)
x11()
ggsurvplot(fml_KM,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability (percentage)",xlab = " Time (Months)",
           legend.title="Testing cohort")

# VALID KM
val_df$CPoints<-val_df$points
val_df$CPoints[val_df$CPoints <cutpoint] <- 0
val_df$CPoints[val_df$CPoints >=cutpoint] <- 1
# write.xlsx(val_df, 'D:\\Data analysis\\Survival_analysis\\SYSU_Cancer_HCC\\result_sorting\\risk_layered\\clinical_val.xlsx')

fml_KM<-survfit(ExtSurv~CPoints,data=val_df)
x11()
ggsurvplot(fml_KM,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability (percentage)",xlab = " Time (Months)",
           legend.title="Validation cohort")


# clinic+dl
#nomogram

survival = Survival(f_multi)
survival_5 = function(x) survival(24, x)
nom <- nomogram(f_multi,fun=list(survival_5),
                funlabel = c("5 year survival"),
                fun.at = c(0.05,seq(0.1,0.9,by=0.05),0.95), lp=FALSE)
plot(nom)
summary(f_multi)
#risk score
results <- formula_rd(nomogram = nom)
test_df$points <- points_cal(formula = results$formula,rd=test_df)
train_df$points <- points_cal(formula = results$formula,rd=train_df)
val_df$points <- points_cal(formula = results$formula,rd=val_df)

res.cut <- surv_cutpoint(train_df, time="time_event",event="label_event",variables =c("points"))
cutpoint<-res.cut$cutpoint[1,1]

#Train KM
train_df$CPoints<-train_df$points
train_df$CPoints[train_df$CPoints <=cutpoint] <- 0
train_df$CPoints[train_df$CPoints >cutpoint] <- 1
# write.xlsx(train_df, 'D:\\Data analysis\\Survival_analysis\\SYSU_Cancer_HCC\\result_sorting\\risk_layered\\clinical_dl_train.xlsx')

fml_KM1<-survfit(TrainSurv~CPoints,data=train_df)
x11()
ggsurvplot(fml_KM1,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability (percentage)",xlab = " Time (Months)",
           legend.title="Training cohort")

# TEST KM
test_df$CPoints<-test_df$points
test_df$CPoints[test_df$CPoints <cutpoint] <- 0
test_df$CPoints[test_df$CPoints >=cutpoint] <- 1
# write.xlsx(test_df, 'D:\\Data analysis\\Survival_analysis\\SYSU_Cancer_HCC\\result_sorting\\risk_layered\\clinical_dl_test.xlsx')

fml_KM<-survfit(TestSurv~CPoints,data=test_df)
x11()
ggsurvplot(fml_KM,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability (percentage)",xlab = " Time (Months)",
           legend.title="Testing cohort")

# VALID KM
val_df$CPoints<-val_df$points
val_df$CPoints[val_df$CPoints <cutpoint] <- 0
val_df$CPoints[val_df$CPoints >=cutpoint] <- 1
# write.xlsx(val_df, 'D:\\Data analysis\\Survival_analysis\\SYSU_Cancer_HCC\\result_sorting\\risk_layered\\clinical_dl_val.xlsx')

fml_KM<-survfit(ExtSurv~CPoints,data=val_df)
x11()
ggsurvplot(fml_KM,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability (percentage)",xlab = " Time (Months)",
           legend.title="Validation cohort")

# dl
#Train KM
#nomogram

survival = Survival(f_dl)
survival_5 = function(x) survival(24, x)
nom <- nomogram(f_dl,fun=list(survival_5),
                funlabel = c("5 year survival"),
                fun.at = c(0.05,seq(0.1,0.9,by=0.05),0.95), lp=FALSE)
plot(nom)
summary(f_multi)

#risk score
results <- formula_rd(nomogram = nom)

res.cut <- surv_cutpoint(train_df, time="time_event",event="label_event",variables =c("dl_ft"))
cutpoint<-res.cut$cutpoint[1,1]

#Train KM
train_df$CPoints<-train_df$dl_ft
train_df$CPoints[train_df$CPoints <=cutpoint] <- 0
train_df$CPoints[train_df$CPoints >cutpoint] <- 1
# write.xlsx(train_df, 'D:\\Data analysis\\Survival_analysis\\SYSU_Cancer_HCC\\result_sorting\\risk_layered\\dl_train.xlsx')

fml_KM1<-survfit(TrainSurv~CPoints,data=train_df)
x11()
ggsurvplot(fml_KM1,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability (percentage)",xlab = " Time (Months)",
           legend.title="Training cohort")


#Test KM
test_df$CPoints<-test_df$dl_ft
test_df$CPoints[test_df$CPoints <=cutpoint] <- 0
test_df$CPoints[test_df$CPoints >cutpoint] <- 1
# write.xlsx(test_df, 'D:\\Data analysis\\Survival_analysis\\SYSU_Cancer_HCC\\result_sorting\\risk_layered\\dl_test.xlsx')

fml_KM1<-survfit(TestSurv~CPoints,data=test_df)
x11()
ggsurvplot(fml_KM1,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability (percentage)",xlab = " Time (Months)",
           legend.title="Testing cohort")


#Valid KM
val_df$CPoints<-val_df$dl_ft
val_df$CPoints[val_df$CPoints <=cutpoint] <- 0
val_df$CPoints[val_df$CPoints >cutpoint] <- 1
# write.xlsx(val_df, 'D:\\Data analysis\\Survival_analysis\\SYSU_Cancer_HCC\\result_sorting\\risk_layered\\dl_val.xlsx')

fml_KM1<-survfit(ExtSurv~CPoints,data=val_df)
x11()
ggsurvplot(fml_KM1,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability (percentage)",xlab = " Time (Months)",
           legend.title="Validation cohort")

