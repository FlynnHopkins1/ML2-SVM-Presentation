rm(list=ls())

options(scipen = 999)
library(e1071)


##6A
#Generate two-class data with p = 2 in such a way that the classes
#are just barely linearly separable.


set.seed(1)
x <- matrix(rnorm(2000 * 2), ncol = 2)
y <- c(rep(-1, 1000), rep(1, 1000))
x[y == 1, ] <- x[y == 1, ] + 5
plot(x, col = (y + 5) / 2, pch = 19)
#dat <- data.frame(x = x, y = as.factor(y))


set.seed(1)
x <- matrix(rnorm(10000 * 2), ncol = 2)
y <- c(rep(-1, 5000), rep(1, 5000))
x[y == 1, ] <- x[y == 1, ] + 5
plot(x, col = (y + 5) / 2, pch = 19)
dat <- data.frame(x = x, y = as.factor(y))


##6B
#Compute the cross-validation error rates for support vector
#classifiers with a range of cost values. How many training
#errors are misclassified for each value of cost considered, and how
#does this relate to the cross-validation errors obtained?

costs <- c(0.001,0.01,0.1,1,10,100,1000,10000)
svmfit <- svm(y ~ ., data = dat , kernel = "linear", cost=10)
svmfit$index
summary(svmfit)
plot(svmfit, dat)


tune.fit.train <- tune(svm, y ~., data = dat, kernel = 'linear', ranges = list(cost = costs))
summary(tune.fit.train)


bestmod <- tune.fit.train$best.model
summary(bestmod)

ypred <- predict(bestmod, dat)
table(predict = ypred, truth = dat$y)




##6C
#Generate an appropriate test data set, and compute the test
#errors corresponding to each of the values of cost considered.
#Which value of cost leads to the fewest test errors, and how
#does this compare to the values of cost that yield the fewest
#training errors and the fewest cross-validation errors?


xtest <- matrix(rnorm (500 * 2), ncol = 2)
ytest <- sample(c(-1, 1), 500, rep = TRUE)
xtest[ytest == 1, ] <- xtest[ytest == 1, ] + 1
testdat <- data.frame(x = xtest, y = as.factor(ytest))


tune.fit.test <- tune(svm, y ~., data = testdat, kernel = 'linear', ranges = list(cost = costs))
summary(tune.fit.train)

bestmod <- tune.fit.test$best.model
summary(bestmod)

ypred <- predict(bestmod, testdat)
table(predict = ypred, truth = testdat$y)


##6D
#Discuss your results.


#What we see is the bias/variance tradeoff.
#As the cost is increased, the model becomes more ‘flexible’ and the training error goes down.
#With the test data, as the flexibility increases we see a decrease in errors until it reaches an inflection point.
#After this point the model is over fitting and the test error increases.
