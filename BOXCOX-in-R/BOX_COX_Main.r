library(MASS)

rm(list=ls())

housing_data = scan('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')
housing_df = as.data.frame(matrix(housing_data, ncol=14, byrow=TRUE), stringsAsFactors=FALSE)
housing_prices = housing_df[, c(14)]
features = housing_df[, c(1:13)]
housing_reg = lm(housing_prices~as.matrix(features)) 
par(mfrow=c(2,2))
plot(housing_reg)

orig_fitted_values = fitted(housing_reg)
par(mfrow=c(1,1))
orig_std_res = stdres(housing_reg)
plot(orig_fitted_values, orig_std_res, main="Fitted Values vs Standardized Residuals", xlab="Fitted Values"
     ,ylab="Standardized Residuals")

outlier_list_extreme_stand_res = c(369, 372, 373)

features_v2 = features[-outlier_list_extreme_stand_res, ]
housing_prices_v2 = housing_prices[-outlier_list_extreme_stand_res]
housing_reg_v2 = lm(housing_prices_v2~as.matrix(features_v2))
par(mfrow=c(2,2))
plot(housing_reg_v2)

second_outlier_list = c(369, 368, 366)

features_v3 = features_v2[-second_outlier_list, ]
housing_prices_v3 = housing_prices_v2[-second_outlier_list]
housing_reg_v3 = lm(housing_prices_v3~as.matrix(features_v3))
par(mfrow=c(2,2))
plot(housing_reg_v3)

# third_outlier_list = c(407)
third_outlier_list = c(375)
features_v4 = features_v3[-third_outlier_list, ]
housing_prices_v4 = housing_prices_v3[-third_outlier_list]
housing_reg_v4 = lm(housing_prices_v4~as.matrix(features_v4))
par(mfrow=c(2,2))
plot(housing_reg_v4)

final_outlier_list = c(406)

features_v5 = features_v4[-final_outlier_list, ]
housing_prices_v5 = housing_prices_v4[-final_outlier_list]
housing_reg_v5 = lm(housing_prices_v5~as.matrix(features_v5))
par(mfrow=c(2,2))
plot(housing_reg_v5)

library(prodlim)
par(mfrow=c(1,1))
matching_rows = row.match(features, features_v5)
outlier_indices = which(is.na(matching_rows))
print(outlier_indices)

# run the box-cox transformation
bc <- boxcox(housing_prices_v5~as.matrix(features_v5)) 

# find the best parameter 
(lambda <- bc$x[which(bc$y==max(bc$y))])  

# transforamting the dependant variable 
new_dep_var = ((((housing_prices_v5)^lambda)-1)/lambda)
# now apply regression model again 
afterboxcox <- lm(new_dep_var ~ (as.matrix(features_v5))) 

par(mfrow=c(2,2))
plot(afterboxcox)

# plotting the fitted house price against the true price  
par(mfrow=c(1,1))
# plot(new_dep_var, housing_prices_v4)

## TODO is to get the predicted value using the last model ..
stand_red_after_box_cox = stdres(afterboxcox)
fitted_box_cox_vals = predict(afterboxcox)
# reverted_fitted_box_cox_vals = 10^(log10(fitted_box_cox_vals*lambda + 1)/lambda)
reverted_fitted_box_cox_vals = (fitted_box_cox_vals*lambda + 1)^(1/lambda)
plot(reverted_fitted_box_cox_vals, housing_prices_v5, main="Fitted Values vs Actual housing prices",
     xlab="Fitted Values", ylab="Housing Prices")
plot(reverted_fitted_box_cox_vals, stand_red_after_box_cox, main="Fitted values vs Standardized residuals",
     xlab="Fitted Values", ylab="Standardized Residuals")