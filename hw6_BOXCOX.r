
library(MASS)
d = read.table("housing.txt")
multi.fit = lm(d$V14~d$V1+d$V2+d$V3+d$V4+d$V5+d$V6+d$V7+d$V8+d$V9+d$V10+d$V11+d$V12+d$V13, data=d) 
plot(multi.fit$residuals)
# run the box-cox transformation
bc <- boxcox(d$V14~d$V1+d$V2+d$V3+d$V4+d$V5+d$V6+d$V7+d$V8+d$V9+d$V10+d$V11+d$V12+d$V13)
# find the best lambda value 
# (lambda1 <- bc$x[which.max(bc$y)])  
(lambda <- bc$x[which(bc$y==max(bc$y))])  

# transforamtion after applying boxcox 
postprocessed <- lm((((d$V14)^lambda-1)/lambda) ~ (d$V1+d$V2+d$V3+d$V4+d$V5+d$V6+d$V7+d$V8+d$V9+d$V10+d$V11+d$V12+d$V13)) # Instead of mnew <- lm(y^trans ~ x) 
# summary(postprocessed) 
plot(postprocessed)
