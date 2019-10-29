# ML
ML files for widely known methods and best practices in Python and R
CS 498 AML - HW6
Team members: Sooraj Kumar, Mostafa Elkabir

Indices of data points in the housing data set that was removed:

366, 368, 369, 370, 372, 373, 381, 413

BoxCox Curve:


By extracting the value of lambda that produces the largest log-likelihood the best lambda value we received was 0.2626263. This is within the 95% confidence interval for the optimal value of lambda as seen in the curve above.

















Diagnostic plots for identifying outliers:

1. Diagnostic plots showing residuals vs leverage vs cooks distance before removing outliers.

2. The points with very high standardized residuals and high leverage are 369, 372, 373 and are removed


3. Again, we remove the points shown in the graph noted for having high standardized residuals and high leverage: 369, 368, 366

4. The point taken then is 375 as seen in the above graph. It was removed due to extreme leverage

5. The last point taken was 406 as it would produce an optimal value of lambda of 0.26













Standardized residuals vs fitted values

1. Standardized residuals vs fitted values before all outliers are removed

2. Standardized residuals vs fitted values after all outliers are removed and transformation

The extreme standardized residuals that are shown at the top of the first graph are completely removed. All standardized residuals fall within 4 and -4. This makes sense as the remaining standardized residuals after all outliers were removed also lay in this range except for one extreme data point that has a standardized residual with a value close to 6. That data point was removed due to the transformation.
Final plot of Fitted house price vs True house price

As seen in the above graph, there is a strong linear relationship between the fitted values and the housing prices. This means that the linear model calculated by the linear regression is quite accurate. Indeed, the plot seems to be closely imitating a 45 degree line from the origin which is a very good sign that the regression model is making close to correct predictions. The removal of outliers means that the line (and thus values of the weights or slope) are not skewed or distorted due to extreme values. 
















Code Snippets:  

In this code we apply BOX-COX transformation to dependent variable after we removed all the outliers. The independent variable is in our case named housing_prices_v4, then we calculated lambda by locating the idx where the log likelihood is maximized. Then used it to transform the dependent variable and apply regression model over the new set, when plotting we also made sure to bring the predicted output of the model back to the original domain by undoing the transformation.


				   COXBOX Transformation 

The following code show all linear regressions performed in the code aside from the regression of the transformed dependent variable and the independent variables. Namely, the linear regression to identify and remove outliers. Code for identifying the indices removed from the original data set is also included.














			
