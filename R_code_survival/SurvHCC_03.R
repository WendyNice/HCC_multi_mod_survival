library(readxl)
library("survival")
library("survminer")
library("ggplot2")
library("ggpubr")
library(rms)
library(plyr)
#install.packages("pec")
library(pec)
#建立虚拟内存
memory.limit(102400)
#读取数据，及数据转换
Mydata <- read_xlsx ("D:/Philips/Hospital/SYSUCC/HCC/survival_analysis_category_eng.xlsx",sheet=1)
Mydata$BCLC<-factor(Mydata$BCLC,ordered = T,levels = c('0','A','B','C'))
Mydata$grading<-factor(Mydata$grading,ordered = T,levels = c('A','B'))
Mydata<-as.data.frame(lapply(Mydata,as.numeric))

#讲数据7：3分为训练和测试集
library(scorecard)
set.seed(20180808)
Data_list <- split_df(Mydata, ratio = 0.7)
TrainData <- Data_list$train
TestData <- Data_list$test

#训练集特征提取
#单因素cox回归
BaSurv<-Surv(time=TrainData$OS,event =TrainData$status)
Unicox<-function(x){
  FML<-as.formula(paste0('BaSurv~',x))
  Sigcox<-coxph(FML,TrainData)
  SigSum<-summary(Sigcox)
  HR<-round(SigSum$coefficients[,2],2)
  Pvalue<-round(SigSum$coefficients[,5],3)
  CI<-paste0(round(SigSum$conf.int[,3:4],2),collapse = '-')
  Unicox<-data.frame('Characteristics'=x,'Hazard Ratio'=HR,'CI95'=CI,'P Value'=Pvalue)
  return(Unicox)
}
Varnames<-colnames(TrainData)[c(1:(ncol(TrainData)-4))]
Univar<-lapply(Varnames,Unicox)
Univar<-ldply(Univar,data.frame)
SigCox_result=Univar[which(Univar$P.Value<0.05),]
write.csv(SigCox_result,"D:/Philips/Hospital/SYSUCC/HCC/KManalysis_Trainresult.csv", row.names=F)

#多因素cox回归
Mydata1<-na.omit(TrainData) #去掉缺省值，否则逐步回归会报错
BaSurv1<-Surv(time=Mydata1$OS,event = Mydata1$status)
fml<-as.formula(paste0('BaSurv1~',paste0(Univar$Characteristics[Univar$P.Value<0.05],collapse = '+')))
MultiCox<-coxph(fml,data=Mydata1)
MultiSum<-summary(MultiCox)
MultiName<-as.character(Univar$Characteristics[Univar$P.Value<0.05])
MHR<-round(MultiSum$coefficients[,2],2)
MPvalue<-round(MultiSum$coefficients[,5],3)
MCI<-paste0(round(MultiSum$conf.int[,3:4],2),collapse = '-')
MCox<-data.frame('Characteristics'=MultiName,'Hazard Ratio'=MHR,'CI95'=MCI,'P Value'=MPvalue)

#逐步回归分析
StepResult<-step(MultiCox,direction = 'backward')
StepSum<-summary(StepResult)


#训练集nomogram
fml_survival<-as.formula(paste0('BaSurv1~',paste0(rownames(StepSum$conf.int),collapse = '+')))
ddist <- datadist(Mydata1)
options(datadist='ddist')
f <- cph(fml_survival,data = Mydata1, x=T, y=T, surv=T)
#f <- cph(BaSurv1~sex+PT,data = Mydata1, x=T, y=T, surv=T)
survival = Survival(f)
survival_1 = function(x) survival(12, x)
survival_2 = function(x) survival(60, x)
nom <- nomogram(f,fun=list(survival_1, survival_2),
                lp=FALSE,funlabel = c("1 year survival", "5 year survival"),
                fun.at = c(0.1,seq(0.5,0.9,by=0.1)))
#par(mar=c(2,5,3,2),cex=0.8)##mar 图形空白边界  cex 文本和符号大小
plot(nom)
x11()
#sum_surv_train<- summary(f) 
#c_index <- sum_surv_train$concordance 
cindex_train <- survConcordance(BaSurv1~predict(f,Mydata1))$concordance

#绘制校准曲线
coxm_1 <- cph(fml_survival,data = Mydata1, x=T, y=T, surv=T,time.inc = 60)
cal_1<-calibrate(coxm_1,u=60,cmethod='KM',m=30,B=1000)
par(mar=c(7,4,4,3),cex=1.0)
plot(cal_1,lwd=2,lty=1, ##设置线条形状和尺寸
     errbar.col=c(rgb(0,118,192,maxColorValue = 255)), ##设置一个颜色
     xlab='Nomogram-Predicted Probability of 5-year DFS',#便签
     ylab='Actual 5-year DFS(proportion)',#标签
     col=c(rgb(192,98,83,maxColorValue = 255)),#设置一个颜色
     xlim = c(0,1),ylim = c(0,1)) ##x轴和y轴范围

#测试集
#c_index
BaSurv_test<-Surv(time=TestData$OS,event =TestData$status)
cindex_test <- survConcordance(BaSurv_test~predict(f,TestData))$concordance

t <- c(12,60)
survprob <- predictSurvProb(f,newdata=TestData,times=t)

fml_survival_test<-as.formula(paste0('BaSurv_test~',paste0(rownames(StepSum$conf.int),collapse = '+')))
coxm_1 <- cph(fml_survival_test,data = TestData, x=T, y=T, surv=T,time.inc = 60)
cal_test<-calibrate(coxm_1,u=60,cmethod='KM',m=30,B=1000)
par(mar=c(7,4,4,3),cex=1.0)
plot(cal_test,lwd=2,lty=1, ##设置线条形状和尺寸
     errbar.col=c(rgb(0,118,192,maxColorValue = 255)), ##设置一个颜色
     xlab='Nomogram-Predicted Probability of 5-year DFS',#便签
     ylab='Actual 5-year DFS(proportion)',#标签
     col=c(rgb(192,98,83,maxColorValue = 255)),#设置一个颜色
     xlim = c(0,1),ylim = c(0,1)) ##x轴和y轴范围
