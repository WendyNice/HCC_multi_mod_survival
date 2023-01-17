# feature_selection

# OS Train
train_df <- read.csv ("D:/Data analysis/Survival_analysis/SYSU_Cancer_HCC/result_sorting/multi_variants/train_df_os.csv")
test_df <- read.csv ("D:/Data analysis/Survival_analysis/SYSU_Cancer_HCC/result_sorting/multi_variants/test_df_os.csv")

train_df$label_event <- as.numeric(train_df$label_event)
train_df$time_event <- as.numeric(train_df$time_event)

test_df$label_event <- as.numeric(test_df$label_event)
test_df$time_event <- as.numeric(test_df$time_event)

# Cox proportional hazards regression 
cox_f <- coxph(Surv(time_event, label_event) ~ BCLC+gender+age+Albumin+bilirubin+AFP, data = train_df)    
summary(cox_f)

##############################
# RFS Train
train_df <- read.csv ("D:/Data analysis/Survival_analysis/SYSU_Cancer_HCC/result_sorting/multi_variants/train_df_rfs.csv")
test_df <- read.csv ("D:/Data analysis/Survival_analysis/SYSU_Cancer_HCC/result_sorting/multi_variants/test_df_rfs.csv")

train_df$label_event <- as.numeric(train_df$label_event)
train_df$time_event <- as.numeric(train_df$time_event)

test_df$label_event <- as.numeric(test_df$label_event)
test_df$time_event <- as.numeric(test_df$time_event)

# Cox proportional hazards regression 
cox_f <- coxph(Surv(time_event, label_event) ~ BCLC+gender+age+Albumin+bilirubin+AFP, data = train_df)    
summary(cox_f)
