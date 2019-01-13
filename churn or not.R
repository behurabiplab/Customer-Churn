######################################Customer Churn#################################################

#Reaading the required dataset
rm(list=ls())

train=read.csv(file = 'C:/Users/user/Desktop/Project/Churn or not/Train_data.csv',stringsAsFactors = FALSE)
test=read.csv('C:/Users/user/Desktop/Project/Churn or not/Test_data.csv',stringsAsFactors = FALSE)
View(train)
dim(train)
#[1] 3333   20
dim(test)
#[1] 1667   20
sum(is.na(train))#there is no missing value
[1] 0
prop.table(table(train$Churn))
# False.     True. 
#0.8550855 0.1449145  
#so there is class imbalance
colnames(train)
#[1] "state"                         "account.length"                "area.code"                    
#[4] "phone.number"                  "international.plan"            "voice.mail.plan"              
#[7] "number.vmail.messages"         "total.day.minutes"             "total.day.calls"              
#[10] "total.day.charge"              "total.eve.minutes"             "total.eve.calls"              
#[13] "total.eve.charge"              "total.night.minutes"           "total.night.calls"            
#[16] "total.night.charge"            "total.intl.minutes"            "total.intl.calls"             
#[19] "total.intl.charge"             "number.customer.service.calls" "Churn" 

str(train)
# $ state                        : chr  "KS" "OH" "NJ" "OH" ...
# $ account.length               : int  128 107 137 84 75 118 121 147 117 141 ...
#$ area.code                    : int  415 415 415 408 415 510 510 415 408 415 ...
#$ phone.number                 : chr  " 382-4657" " 371-7191" " 358-1921" " 375-9999" ...
#$ international.plan           : chr  " no" " no" " no" " yes" ...
#$ voice.mail.plan              : chr  " yes" " yes" " no" " no" ...
#$ number.vmail.messages        : int  25 26 0 0 0 0 24 0 0 37 ...
#$ total.day.minutes            : num  265 162 243 299 167 ...
#$ total.day.calls              : int  110 123 114 71 113 98 88 79 97 84 ...
#$ total.day.charge             : num  45.1 27.5 41.4 50.9 28.3 ...
#$ total.eve.minutes            : num  197.4 195.5 121.2 61.9 148.3 ...
#$ total.eve.calls              : int  99 103 110 88 122 101 108 94 80 111 ...
#$ total.eve.charge             : num  16.78 16.62 10.3 5.26 12.61 ...
#$ total.night.minutes          : num  245 254 163 197 187 ...
#$ total.night.calls            : int  91 103 104 89 121 118 118 96 90 97 ...
#$ total.night.charge           : num  11.01 11.45 7.32 8.86 8.41 ...
#$ total.intl.minutes           : num  10 13.7 12.2 6.6 10.1 6.3 7.5 7.1 8.7 11.2 ...
#$ total.intl.calls             : int  3 3 5 7 3 6 7 6 4 5 ...
#$ total.intl.charge            : num  2.7 3.7 3.29 1.78 2.73 1.7 2.03 1.92 2.35 3.02 ...
#$ number.customer.service.calls: int  1 1 0 2 3 0 3 0 1 0 ...
#$ Churn                        : chr  " False." " False." " False." " False." ...



#Dropping the phone number column
train=train[,-4]
test=test[,-4]
View(train)

#converting variables to factor_variables
str(train)
factor_var=c('state','international.plan','voice.mail.plan','Churn','area.code')

#converting variable to factor variable in both train and test data
for (i in factor_var){
  train[,which(colnames(train)==i)]=as.factor(train[,which(colnames(train)==i)])
}


for (i in factor_var){
  test[,which(colnames(test)==i)]=as.factor(test[,which(colnames(test)==i)])
}

summary(train)

#plotting churn 
library(ggplot2)
ggplot(data = train,aes(x = Churn,fill = Churn))+
  geom_bar() +  labs(y='Churn Count', title = 'Customer Churn or Not')



#plottingcchurn according to state
ggplot(data = train,aes(x=state,fill = Churn))+
  geom_bar() +  labs(y='Churn Count', title = 'Customer Churn or Not')
factor_var

#plotting churn according to international plan
ggplot(data = train,aes(x=international.plan,fill = Churn))+
  geom_bar() +  labs(y='Churn Count', title = 'Customer Churn or Not')

#plotting churn according to voice mail plan
ggplot(data = train,aes(x=voice.mail.plan,fill = Churn))+
  geom_bar() +  labs(y='Churn Count', title = 'Customer Churn or Not')

#plotting churn according to area code
ggplot(data = train,aes(x=area.code,fill = Churn))+
  geom_bar() +  labs(y='Churn Count', title = 'Customer Churn or Not')

#Churn by area code  group by  voice mail plan
ggplot(train, aes(fill=Churn, x=area.code)) +
  geom_bar(position="dodge") + facet_wrap(~international.plan)+
  labs(title="Churn on Area Code  group by International Plan")


ggplot(train, aes(fill=Churn, x=area.code)) +
  geom_bar(position="dodge") + facet_wrap(~voice.mail.plan)+
  labs(title="Churn on Area Code  group by International Plan")



#assigning levels for factor variable in both train and test dataset
for (i in factor_var){
  train[,i]=factor(train[,i],labels = (1:length(levels(train[,i]))))
}

for (i in factor_var){
  test[,i]=factor(test[,i],labels = (1:length(levels(test[,i]))))
}
View(test)

#Extracting the numeric variable name from the dataframe
numeric_var=colnames(train[sapply(train, is.numeric)])
numeric_var


################outlier analysis#######################

##As random forest require not input prepartio we can skip this step for random forest
##But for logistic regression we have to do outlier analysis


par(mfrow=c(4,4))
for (i in numeric_var){
  boxplot(train[,which(colnames(train)==i)],main=paste('var',i))
}

#imputing outliers with missing value
for(i in numeric_var){
  out=boxplot.stats(train[,which(colnames(train)==i)])$out
  train[train[,which(colnames(train)==i)] %in% out,which(colnames(train)==i)]=NA
}

sum(is.na(train))
colSums(is.na(train))


#imputing missing values with proper method
train[8,'total.eve.minutes']
#103.1
train[8,'total.eve.minutes']=NA
mean(train$total.eve.minutes,na.rm = T)
median(train$total.eve.minutes,na.rm = T)
library(DMwR)
train=knnImputation(train,k=3)
train[8,'total.eve.minutes']
#Here we check the mean,median and knn i found knn is the best method
#knn imputation value is 114 so it is close to 103. so we choose knn imputation

#extracting the name of the factor variable and excluding the churn variable
factor_var=colnames(train[sapply(train, is.factor)])
factor_var=factor_var[-length(factor_var)]
factor_var



##############################chisquare test################################chisquare test############################
for (i in factor_var){
  print(i)
  print(chisq.test(table(train$Churn,train[,which(colnames(train)==i)])))
}
#so we have to drop area.code
train=train[,-which(colnames(train)=="area.code")]
test=test[,-which(colnames(test)=="area.code")]
#Multicolinearity test

par(mfrow=c(1,1))
x=cor(train[numeric_var])
library(corrplot)
corrplot(x)
library(caret)
corelation=findCorrelation(x,cutoff = 0.7)
corelation
numeric_var
corelation=numeric_var[corelation]
corelation

#dropping the multicolinearity variables
train=train[,-which(colnames(train) %in% corelation)]
test=test[,-which(colnames(test) %in% corelation)]

###########################Normalizing variable##########################Normalizing variable ###########################
library(lattice)
numeric_var=colnames(train[sapply(train, iis.numeric)])
numeric_var

par(mfrow=c(4,3))
for ( i in numeric_var){
  hist(train[,which(colnames(train)==i)],main = paste0('var','_',i))
}



#zscore

numeric_var=colnames(train[sapply(train, is.numeric)])
numeric_var
for (i in numeric_var){
  train[,which(colnames(train)==i)]=scale(train[,which(colnames(train)==i)])
}

for (i in numeric_var){
  test[,which(colnames(test)==i)]=scale(test[,which(colnames(test)==i)])
}


################################Dealing with class imbalance problem#############################
prop.table(table(train$Churn))
library(ROSE)
#train=ovun.sample(Churn~.,data=train,method = "over",N=(2850*2))$data
train=ovun.sample(Churn~.,data = train,method = 'both',p=0.5)$data
table(train$Churn)
#table(both$Churn)

###################Function for confusion matrix###################################################
cf=function(x){
  TN=x[1,1]
  FP=x[1,2]
  FN=x[2,1]
  TP=x[2,2]
  print(paste('accuracy is',(TP+TN)/(TP+TN+FP+FN)))
  print(paste('precission is',(TP/(TP+FP))))
  print(paste('sensitivity is',(TP/(TP+FN))))
  print(paste('specificity is',(TN/(TN+FP))))
  print(paste('false positive rate is',(FP/(FP+TN))))
  print(paste('false negative rate is',(FN)/(FN+TP)))
  
}


########################logistic regression model######################################################
log_model=glm(Churn~.,data = train,family = 'binomial')
summary(log_model)
predict=predict(log_model,test,type='response')
table(train$Churn)
confusion_matrix=table(ActualValue=test$Churn,
                       PredictedValue=predict>0.5)
confusion_matrix


cf(confusion_matrix)

new_predict=ifelse(predict > 0.5,2,1)


#now lets increse the prob
new_predict=ifelse(predict > 0.6,2,1)

library(ROCR)
prediction=prediction(predict,test$Churn)
perf=performance(prediction,"tpr","fpr")
plot(perf)
abline(0,1,lty=2)


############random forest#######################random forest###################

library(randomForest)
rf=randomForest(Churn~.,data = train,ntree=500)
rf
rf
summary(rf)

plot(rf)

#tuning random forest
mtry <- tuneRF(train[-(ncol(train))],train$Churn, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)

mtry
best_mtry=mtry[mtry[,2]==min(mtry[,2]),1]
best_mtry

set.seed(3424)
rf1=randomForest(Churn~.,data=train,mtry=best_mtry,importance=TRUE,ntree=500)
print(rf1)

predict=predict(rf1,newdata=test)
confusion_matrix=table(test$Churn,predict)
cf(confusion_matrix)


################knn##########################################knn############################ 
knn_train=train[,-which(colnames(train)=='Churn')]
knn_test=test[,-which(colnames(test)=='Churn')]
train_labels=train[,which(colnames(train)=='Churn')]
test_labels=test[,which(colnames(test)=='Churn')]
library(class)
knnmodel=knn(train=knn_train,test=knn_test,cl=train_labels,k=5)
confusion_matrix=table(test$Churn,knnmodel)
cf(confusion_matrix)
knnmodel2=knn(train=knn_train,test=knn_test,cl=train_labels,k=7)
confusion_matrix=table(test$Churn,knnmodel2)
cf(confusion_matrix)

knnmodel3=knn(train=knn_train,test=knn_test,cl=train_labels,k=3)
confusion_matrix=table(test$Churn,knnmodel)
cf(confusion_matrix)


#######################################naive bayes###################################
library(e1071)
model1 = naiveBayes(Churn ~., data = train, type = 'class')
predict=predict(model1,newdata=test)
confusion_matrix=table(test$Churn,predict)
cf(confusion_matrix)
save.image()



