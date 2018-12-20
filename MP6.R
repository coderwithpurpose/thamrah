rm(list=ls())

housing_data = scan('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')
housing_df = as.data.frame(matrix(housing_data, ncol=14, byrow=TRUE), stringsAsFactors=FALSE)
housing_prices = housing_df[, c(14)]
features = housing_df[, c(1:13)]

housing_reg = lm(housing_prices~as.matrix(features))
par(mfrow=c(2,2))
plot(housing_reg)

cooksd_housing = cooks.distance(housing_reg)
par(mfrow=c(1,1))
plot(cooksd_housing, cex=1, main="Finding outliers via Cooks distance")
abline(h = 4*mean(cooksd_housing, na.rm=T), col="red")
text(x=1:length(cooksd_housing)+1, y=cooksd_housing, labels=ifelse(cooksd_housing>2*mean(cooksd_housing, na.rm=T),names(cooksd_housing),""), col="red")

outliers_log_vec = !(cooksd_housing > 2*mean(cooksd_housing, na.rm=T))
features_v2 = features[outliers_log_vec, ]
housing_prices_v2 = housing_prices[outliers_log_vec]

housing_reg_v2 = lm(housing_prices_v2~as.matrix(features_v2))
par(mfrow=c(2,2))
# plot(housing_reg_v2)

library(MASS)
stand_red = stdres(housing_reg_v2)
extreme_res = !((stand_red > 3) | (stand_red < -3))
features_v3 = features_v2[extreme_res, ]
housing_prices_v3 = housing_prices_v2[extreme_res]

housing_reg_v3 = lm(housing_prices_v3~as.matrix(features_v3))
par(mfrow=c(2,2))
plot(housing_reg_v3)

threshold_cooks_d = 0.10
features_v3_mat = as.matrix(features_v3)
hat_matrix = features_v3_mat%*%solve(t(features_v3_mat) %*% features_v3_mat)%*%t(features_v3_mat)
extreme_lev_d_log_vec = !(diag(hat_matrix)  > threshold_cooks_d)
features_v4 = features_v3[extreme_lev_d_log_vec, ]
housing_prices_v4 = housing_prices_v3[extreme_lev_d_log_vec]

housing_reg_v4 = lm(housing_prices_v4~as.matrix(features_v4))
par(mfrow=c(2,2))
plot(housing_reg_v4)



# 
# features_v2_mat = as.matrix(features_v2)
# hat_matrix = features_v2_mat%*%solve(t(features_v2_mat) %*% features_v2_mat)%*%t(features_v2_mat)
# high_lev_points = tail(sort(diag(hat_matrix), index.return=TRUE)$ix)
# 
# features_v3 = features_v2[-high_lev_points, ]
# housing_prices_v3 = housing_prices_v2[-high_lev_points]
# 
# housing_reg_v3 = lm(housing_prices_v3~as.matrix(features_v3))
# par(mfrow=c(2,2))
# plot(housing_reg_v3)
# 
# library(MASS)
# stand_red = stdres(housing_reg_v3)
# extreme_res = !((stand_red > 3) | (stand_red < -3))
# features_v4 = features_v3[extreme_res, ]
# housing_prices_v4 = housing_prices_v3[extreme_res]
# 
# housing_reg_v4 = lm(housing_prices_v4~as.matrix(features_v4))
# par(mfrow=c(2,2))
# plot(housing_reg_v4)
# 
# extreme_cooks_d = tail(sort(cooks.distance(housing_reg_v4), index.return=TRUE)$ix)

# features_v2 = features[c(-369, -372, -373, -365),]
# housing_prices_v2 = housing_prices[c(-369, -372, -373, -365)]
# housing_reg_v2 = lm(housing_prices_v2~as.matrix(features_v2))
# par(mfrow=c(2,2))
# plot(housing_reg_v2)
# 
# features_v3 = features_v2[c(-365, -368, -369, -367),]
# housing_prices_v3 = housing_prices_v2[c(-365, -368, -369, -367)]
# housing_reg_v3 = lm(housing_prices_v3~as.matrix(features_v3))
# par(mfrow=c(2,2))
# plot(housing_reg_v3)

