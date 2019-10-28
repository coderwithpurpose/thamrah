
library(MASS)
# library( gdata )
library(glmnet)  
library(tidyverse)
rm(list=ls())

d = read.csv('Crusio1.csv')
test = (d[,1:41])  
test = test %>% drop_na()
new_d = within(test,sex  <- factor(sex, labels = c(0, 1)))  
int_labels = as.matrix(new_d$sex)  
ymat<−as.matrix(new_d[,c(2)]) 
xmat<−as.matrix( new_d[, c(4:41)])  

# f is 0, m is 1 
# Note alpha=1 for lasso only and can blend with ridge penalty down to
# alpha=0 ridge only.  // Binomial model to predict gender ogf mouse 
par(mfrow=c(2 ,3))
fit <- glmnet(x=xmat,y=ymat , alpha=1 ,family="binomial")
plot(fit)
# plot(fit, xvar = "dev", label = TRUE)
lasso_model <- cv.glmnet(x=xmat,y=ymat , alpha=1 ,family="binomial", type.measure = "class") 
plot(lasso_model) 
predict = predict(lasso_model, newx = xmat, s = "lambda.min", type = "class") 

# multinmoial using lasso but now to predict strain
fit = glmnet(x=xmat,y=ymat,alpha = 1, family = "multinomial") #type.multinomial = "grouped") 
plot(fit)
lasso_model2 <- cv.glmnet(x=xmat,y=ymat , alpha=1 ,family="multinomial", type.multinomial = "grouped", parallel = TRUE) 
plot(lasso_model2) 
predict(lasso_model2, newx = xmat[1:10,], s = "lambda.min", type = "class") 
