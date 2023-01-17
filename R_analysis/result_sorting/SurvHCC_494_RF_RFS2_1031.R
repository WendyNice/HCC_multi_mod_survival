rm(list=ls())
#install.packages("glmnet")
#install.packages("survcomp")
library(glmnet)
library(readxl)
library("survival")
library("survminer")
library("ggplot2")
library("ggpubr")
library(rms)
library(plyr)
#install.packages("pec")
library(pec)
#install.packages("randomForestSRC")
library(randomForestSRC)
#install.packages("nomogramFormula")
library(nomogramFormula)
#install.packages("rmda")
library(rmda)
#install.packages("survivalROC")
library(survivalROC)
library(survcomp)
#建立虚拟内存
memory.limit(102400)

#读取数据，及数据转换
Mydata <- read_xlsx ("E:/Philips/Hospital/SYSUCC/HCC/494_0708/0818/494_0824_BCLCmodify.xlsx",sheet='Sheet1')
Mydata <- Mydata[,9:ncol(Mydata)]
ExtValidData<-read_xlsx ("E:/Philips/Hospital/SYSUCC/HCC/followup3.xlsx",sheet='Sheet2')
#有序分类变量
Mydata$BCLC<-factor(Mydata$BCLC,ordered = T,levels = c('0','A','B','C'))
Mydata$differentiation<-factor(Mydata$differentiation,ordered = T,levels = c('1','2','3'))
Mydata$grading<-factor(Mydata$grading,ordered = T,levels = c('A','B'))
ExtValidData$BCLC<-factor(ExtValidData$BCLC,ordered = T,levels = c('0','A','B','C'))

#连续变量归一化
#Mydata[,c(2,22)]<-as.data.frame(lapply(Mydata[,c(2,22)],as.numeric))
#Mydata[,c(2,22)]<-scale(Mydata[,c(2,22)], center = T, scale = T)
#Mydata[,-c(17,19,20)]<-as.data.frame(lapply(Mydata[,-c(17,19,20)],as.numeric))

Mydata<-as.data.frame(lapply(Mydata,as.numeric))
ExtValidData<-as.data.frame(lapply(ExtValidData,as.numeric))

#数据按手术入组时间7：3分为训练和测试集
#library(scorecard)
#set.seed(20180808)
#Data_list <- split_df(Mydata, ratio = 0.7)
TrainData <- Mydata[1:346,]
TestData <- Mydata[347:nrow(Mydata),]
TrainData<-na.omit(TrainData)
TestData<-na.omit(TestData)

months=24;
years=2; 

#训练集特征提取RF
#更改因变量
Traintime<-TrainData$PFS_two;Trainevent<-TrainData$Progress_two
Testtime<-TestData$PFS_two;Testevent<-TestData$Progress_two
Exttime<-ExtValidData$PFS_two;Extevent<-ExtValidData$Progress_two

TrainSurv<-Surv(Traintime,Trainevent)
TestSurv<-Surv(Testtime,Testevent)
ExtSurv<-Surv(Exttime,Extevent)

TrainLabel<-as.matrix(TrainSurv)
TrainFeatures<-as.matrix(TrainData[,1:(ncol(TrainData)-12)])


# d<-0
# i=0
# while(i<100){
#   i=i+1
#   #fit <-glmnet(TrainFeatures,TrainLabel,family = "cox",alpha = 1)
#   #x11()
#   #plot(fit,xvar="lambda",label=T)
#   
#   #主要在做交叉验证,lasso
#   fitcv <- cv.glmnet(TrainFeatures,TrainLabel,family="cox", alpha=1,nfolds=10)
#   #x11()
#   #plot(fitcv)
#   fitcv_coef<-coef(fitcv, s="lambda.min")
#   select_varialbes = rownames(as.data.frame(which(fitcv_coef[,1]!=0)))
#   d<-append(d,select_varialbes)
# }
# 
# d=d[-1]  #remove 0
# #d
# length(d)
# d=as.data.frame(table(d))
# d=d[order(d[,"Freq"],decreasing=TRUE),]
# d2=d[d$Freq==100,]#选取频率最高的特征
# 
# d<-0
# i=0
# while(i<100){
#   i=i+1
#   ##Surv(PFS_two,Progress_two)~需要更改
#   FML<-as.formula(paste0('Surv(PFS_two,Progress_two)~',paste0(colnames(TrainFeatures),collapse = '+')))
#   RF<- rfsrc(FML, TrainData, nodesize = 20, importance = TRUE)
#   RFvar <- var.select(RF, conservative = "high")
#   topvars <- RFvar$topvars
#   d<-append(d,topvars)
# }
# d=d[-1]  #remove 0
# #d
# length(d)
# d=as.data.frame(table(d))
# d=d[order(d[,"Freq"],decreasing=TRUE),]
# d2=d[d$Freq==100,]#选取频率最高的特征
# fml<-as.formula(paste0('TrainSurv~',paste0(d2$d,collapse = '+')))

fml=TrainSurv ~ age+BCLC+maximum_diameter+microvascular_invasion + 
  nonsmooth_margin+focus
ddist <- datadist(TrainData)
options(datadist='ddist')
options(contrasts=c("contr.treatment", "contr.treatment"))

MultiCox <- cph(fml,data = TrainData, x=T, y=T, surv=T)
f<-step(MultiCox,direction = 'backward')

##加入radscore
# fml<-TrainSurv ~ BCLC+ maximum_diameter + microvascular_invasion + nonsmooth_margin+CRadscore_OS5
# 
# #将radscore改为分类变量
# Radcut <- surv_cutpoint(TrainData, time="PFS_two",event="Progress_two",variables =c("Radscore_OS5"))
# cutpoint<-Radcut$cutpoint[1,1]
# TrainData$CRadscore_OS5<-TrainData$Radscore_OS5
# TrainData$CRadscore_OS5[TrainData$Radscore_OS5<cutpoint] <- 0
# TrainData$CRadscore_OS5[TrainData$Radscore_OS5 >=cutpoint] <- 1
# 
# TestData$CRadscore_OS5<-TestData$Radscore_OS5
# TestData$CRadscore_OS5[TestData$Radscore_OS5 <cutpoint] <- 0
# TestData$CRadscore_OS5[TestData$Radscore_OS5 >=cutpoint] <- 1
# ddist <- datadist(TrainData)
# options(datadist='ddist')
# #options(contrasts=c("contr.treatment", "contr.treatment"))
# f<-cph(fml,data = TrainData, x=T, y=T, surv=T)

#BCLC
BCLCCox_train <- coxph(TrainSurv~BCLC,data = TrainData)

source("C:/Users/320101779/Downloads/downloadrcode/stdca.R")

# aa=rbind(TrainData,TestData)
x11()
ExtValidData$Nomogram=c(1-(summary(survfit(f,ExtValidData),times=24)$surv))
ExtValidData$BCLC_stage=c(1-(summary(survfit(BCLCCox_train,ExtValidData),times=24)$surv))
stdca(data=ExtValidData,outcome="Progress_two",ttoutcome="PFS_two",timepoint=24,predictor=c("Nomogram","BCLC_stage"),
      xstop=.25,smooth=TRUE)

TestData$Nomogram=c(1-(summary(survfit(f,TestData),times=24)$surv))
TestData$BCLC_stage=c(1-(summary(survfit(BCLCCox_train,TestData),times=24)$surv))
stdca(data=TestData,outcome="Progress_two",ttoutcome="PFS_two",timepoint=24,predictor=c("Nomogram","BCLC_stage"),
      xstop=.25,smooth=TRUE)

#nomogram
survival = Survival(f)
survival_1 = function(x) survival(12, x)
survival_2 = function(x) survival(24, x)
#survival_3 = function(x) survival(months, x)
label_pre=cat(years," year survival");
x11()
nom <- nomogram(f,fun=list(survival_1, survival_2),
                lp=FALSE,
                #funlabel = c("1-Year Survival probability","3-Year Survival probability" ,paste(years,"-Year Survival probability")),
                funlabel = c("1-Year Survival probability","2-Year Survival probability"),
                fun.at = c(0.1,seq(0.5,0.9,by=0.1)))

plot(nom)

#第一种算p值的方法:c-index
# cindex_train<-concordance.index(predict(f),surv.time=TrainData$PFS_two,surv.event = TrainData$Progress_two,method = "noether")
# Cindex_test<-concordance.index(predict(f,TestData),surv.time=TestData$PFS_two,surv.event = TestData$Progress_two,method = "noether")
# cindex_extData<-concordance.index(predict(f,ExtValidData),surv.time=ExtValidData$PFS_two,surv.event = ExtValidData$Progress_two,method = "noether")
# 
# BCLC_cindex_train<-concordance.index(predict(BCLCCox_train),surv.time=TrainData$PFS_two,surv.event = TrainData$Progress_two,method = "noether")
# BCLC_cindex_test<-concordance.index(predict(BCLCCox_train,TestData),surv.time=TestData$PFS_two,surv.event = TestData$Progress_two,method="noether")
# BCLC_cindex_extData<-concordance.index(predict(BCLCCox_train,ExtValidData),surv.time=ExtValidData$PFS_two,surv.event = ExtValidData$Progress_two,method = "noether")
# 
# #Cindex, Pvalue
# p_train<-cindex.comp(cindex_train,BCLC_cindex_train)
# p_test<-cindex.comp(Cindex_test,BCLC_cindex_test)
# p_extData<-cindex.comp(cindex_extData,BCLC_cindex_extData)
# 

#第二种计算Cindex的方法
CIN_train <- survConcordance(TrainSurv~predict(f,TrainData))
CIN_test <- survConcordance(TestSurv~predict(f,TestData))
CIN_ext <- survConcordance(ExtSurv~predict(f,ExtValidData))
#置信区间
cindex_train2 <- CIN_train$concordance
cindex_train2$low<-CIN_train$concordance-1.96*CIN_train$std.err
cindex_train2$high<-CIN_train$concordance+1.96*CIN_train$std.err
cindex_test2 <- CIN_test$concordance
cindex_test2$low<-CIN_test$concordance-1.96*CIN_test$std.err
cindex_test2$high<-CIN_test$concordance+1.96*CIN_test$std.err
cindex_ext2 <- CIN_ext$concordance
cindex_ext2$low<-CIN_ext$concordance-1.96*CIN_ext$std.err
cindex_ext2$high<-CIN_ext$concordance+1.96*CIN_ext$std.err

BCLC_CIN_train2 <- survConcordance(TrainSurv~predict(BCLCCox_train,TrainData))
BCLC_cindex_train2 <- BCLC_CIN_train2$concordance
BCLC_cindex_train2$low<-BCLC_CIN_train2$concordance-1.96*BCLC_CIN_train2$std.err
BCLC_cindex_train2$high<-BCLC_CIN_train2$concordance+1.96*BCLC_CIN_train2$std.err

BCLC_CIN_test2 <- survConcordance(TestSurv~predict(BCLCCox_train,TestData))
BCLC_cindex_test2 <- BCLC_CIN_test2$concordance
BCLC_cindex_test2$low<-BCLC_CIN_test2$concordance-1.96*BCLC_CIN_test2$std.err
BCLC_cindex_test2$high<-BCLC_CIN_test2$concordance+1.96*BCLC_CIN_test2$std.err

BCLC_CIN_ext2 <- survConcordance(ExtSurv~predict(BCLCCox_train,ExtValidData))
BCLC_cindex_ext2 <- BCLC_CIN_ext2$concordance
BCLC_cindex_ext2$low<-BCLC_CIN_ext2$concordance-1.96*BCLC_CIN_ext2$std.err
BCLC_cindex_ext2$high<-BCLC_CIN_ext2$concordance+1.96*BCLC_CIN_ext2$std.er

train_pre=list()
train_pre$x=predict(f)
Ctrain=list(c.index=CIN_train$concordance,n=CIN_train$n,se=CIN_train$std.err,data=list())
Ctrain$data=train_pre
BCLC_train_pre=list()
BCLC_train_pre$x=predict(BCLCCox_train)
BCLC_Ctrain=list(c.index=BCLC_CIN_train2$concordance,n=BCLC_CIN_train2$n,se=BCLC_CIN_train2$std.err,data=list())
BCLC_Ctrain$data=BCLC_train_pre
p_train2<-cindex.comp(Ctrain,BCLC_Ctrain)

test_pre=list()
test_pre$x=predict(f,TestData)
Ctest=list(c.index=CIN_test$concordance,n=CIN_test$n,se=CIN_test$std.err,data=list())
Ctest$data=test_pre
BCLC_test_pre=list()
BCLC_test_pre$x=predict(BCLCCox_train,TestData)
BCLC_Ctest=list(c.index=BCLC_CIN_test2$concordance,n=BCLC_CIN_test2$n,se=BCLC_CIN_test2$std.err,data=list())
BCLC_Ctest$data=BCLC_test_pre
p_test2<-cindex.comp(Ctest,BCLC_Ctest)

ext_pre=list()
ext_pre$x=predict(f,ExtValidData)
Cext=list(c.index=CIN_ext$concordance,n=CIN_ext$n,se=CIN_ext$std.err,data=list())
Cext$data=ext_pre
BCLC_ext_pre=list()
BCLC_ext_pre$x=predict(BCLCCox_train,ExtValidData)
BCLC_Cext=list(c.index=BCLC_CIN_ext2$concordance,n=BCLC_CIN_ext2$n,se=BCLC_CIN_ext2$std.err,data=list())
BCLC_Cext$data=BCLC_ext_pre
p_ext2<-cindex.comp(Cext,BCLC_Cext)

#校准曲线
cal_1<-calibrate(f,u=24,cmethod='KM',m=61,B=1000)
x11()
plot(cal_1,lwd=1,lty=1, ##设置线条形状和尺寸
     subtitles = FALSE,
     errbar.col=c(rgb(0,118,192,maxColorValue = 255)), ##设置一个颜色
     xlab='Nomogram-Predicted Probability of 2-Year OS in the training cohort',#便签
     #xlab='Nomogram-Predicted Probability of 2-Year RFS',#便签
     ylab='Actual 2-Year OS (proportion)',#标签
     #ylab=paste("Actual",years,"-Year OS (proportion)"),
     col=c(rgb(192,98,83,maxColorValue = 255)),#设置一个颜色
     xlim = c(0.2,1),ylim = c(0.2,1)) ##x轴和y轴范围


coxm_1 <- cph(TestSurv~predict(f,TestData),data = TestData, x=T, y=T, surv=T)
cal_test<-calibrate(coxm_1,u=24,cmethod='KM',m=30,B=1000)
x11()
plot(cal_test,lwd=2,lty=1, ##设置线条形状和尺寸
     errbar.col=c(rgb(0,118,192,maxColorValue = 255)), ##设置一个颜色
     xlab='Nomogram-Predicted Probability of 2-Year OS',#便签
     ylab='Actual 2-Year OS (proportion)',#标签
     #xlab=paste("Nomogram-Predicted Probability of",years,"-Year OS"),
     #ylab=paste("Actual",years,"-Year OS (proportion)"),
     col=c(rgb(192,98,83,maxColorValue = 255)),#设置一个颜色
     xlim = c(0,1),ylim = c(0,1)) ##x轴和y轴范围

cox_ext <- cph(ExtSurv~predict(f,ExtValidData),data = ExtValidData, x=T, y=T, surv=T)
cal_ext<-calibrate(cox_ext,u=24,cmethod='KM',m=15,B=1000)
x11()
plot(cal_ext,lwd=2,lty=1, ##设置线条形状和尺寸
     errbar.col=c(rgb(0,118,192,maxColorValue = 255)), ##设置一个颜色
     xlab='Predicted probability of 2 years RFS in the external validation cohort',#便签
     ylab='Actual 2 year RFS (proportion)',#标签
     #xlab=paste("Nomogram-Predicted Probability of",years,"-Year OS"),
     #ylab=paste("Actual",years,"-Year OS (proportion)"),
     col=c(rgb(192,98,83,maxColorValue = 255)),#设置一个颜色
     xlim = c(0,1),ylim = c(0,1),subtitles = 0) ##x轴和y轴范围

#risk score
results <- formula_rd(nomogram = nom)
TestData$points <- points_cal(formula = results$formula,rd=TestData)
TrainData$points <- points_cal(formula = results$formula,rd=TrainData)
ExtValidData$points <- points_cal(formula = results$formula,rd=ExtValidData)

res.cut <- surv_cutpoint(TrainData, time="PFS_two",event="Progress_two",variables =c("points"))
cutpoint<-res.cut$cutpoint[1,1]

#Train KM分析
TrainData$CPoints<-TrainData$points
TrainData$CPoints[TrainData$CPoints <cutpoint] <- 0
TrainData$CPoints[TrainData$CPoints >=cutpoint] <- 1
fml_KM1<-survfit(TrainSurv~CPoints,data=TrainData)
x11()
ggsurvplot(fml_KM1,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability (percentage)",xlab = " Time (Months)",
           legend.title="Training cohort")


#test KM 分析
TestData$CPoints<-TestData$points
TestData$CPoints[TestData$CPoints <cutpoint] <- 0
TestData$CPoints[TestData$CPoints >=cutpoint] <- 1
fml_KM<-survfit(TestSurv~CPoints,data=TestData)
x11()
ggsurvplot(fml_KM,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability (percentage)",xlab = " Time (Months)",
           legend.title="Testing cohort")
#ext KM 分析
ExtValidData$CPoints<-ExtValidData$points
ExtValidData$CPoints[ExtValidData$CPoints <cutpoint] <- 0
ExtValidData$CPoints[ExtValidData$CPoints >=cutpoint] <- 1
fml_KM2<-survfit(ExtSurv~CPoints,data=ExtValidData)
x11()
ggsurvplot(fml_KM2,surv.median.line = "hv",conf.int = TRUE,
           legend.labs = c("low-risk", "high-risk"),pval=TRUE,
           ylab="Survival probability",xlab = " Time (Months)",
           legend.title="External validation cohort")
#
TrainRoc=survivalROC(Stime = TrainData$PFS_two,status = TrainData$Progress_two,marker = TrainData$points,predict.time = 60,method = 'KM')
str(TrainRoc)
x11()
plot(TrainRoc$FP,TrainRoc$TP,col='red',xlim=c(0,1),ylim=c(0,1),type = "o",
     xlab = '1-Specificity',ylab='Sensitivity',abline(0,1,col="gray",lty=2))
legend("bottomright",paste("AUC=",round(TrainRoc$AUC,3)))


