####### Team 17: Flynn Hopkins, Dave Maser, TianYu Liu #######

#######TP01#######

#clearing environment variables
rm(list=ls())

#loading scipen for improved quality of life when reading tables
options(scipen = 999)

#calling necessary library for svm and tune
library(e1071)


##9.7.6
#At the end of Section 9.6.1, it is claimed that in the case of data that
#is just barely linearly separable, a support vector classifier with a
#small value of cost that misclassifies a couple of training observations
#may perform better on test data than one with a huge value of cost
#that does not misclassify any training observations. You will now
#investigate this claim.


##6A
#Generate two-class data with p = 2 in such a way that the classes
#are just barely linearly separable.


#setting the seed for consistent data
set.seed(1)

#generating x and y training data using matrix
x <- matrix(rnorm(140 * 2), ncol = 2)
y <- c(rep(-1, 70), rep(1, 70))
x[y == 1, ] <- x[y == 1, ] + 3
plot(x, col = (y + 5) / 2, pch = 19)
train.dat <- data.frame(x = x, y = as.factor(y))



####################################################################



##6B
#Compute the cross-validation error rates for support vector
#classifiers with a range of cost values. How many training
#errors are misclassified for each value of cost considered, and how
#does this relate to the cross-validation errors obtained?



#Computing cross validation error rates for our range of cost values
costs <- c(0.001,0.01,0.1,1,10,100,1000,10000)
tune.fit.train <- tune(svm, y ~., data = train.dat, kernel = "linear", ranges = list(cost = costs))
summary(tune.fit.train)



#Determining how many points were miss classified for each cost value using train data
#Cost = 0.001
svmfit.train1 <- svm(y ~ ., data = train.dat, kernel = "linear", cost=0.001)
pred.train1 <- predict(svmfit.train1, train.dat)
table(predict = pred.train1, truth = train.dat$y)
#3 misclassifictions


#Cost = 0.01
svmfit.train2 <- svm(y ~ ., data = train.dat, kernel = "linear", cost=0.01)
pred.train2 <- predict(svmfit.train2, train.dat)
table(predict = pred.train2, truth = train.dat$y)
#1 misclassification


#Cost = 0.1
svmfit.train3 <- svm(y ~ ., data = train.dat, kernel = "linear", cost=0.1)
pred.train3 <- predict(svmfit.train3, train.dat)
table(predict = pred.train3, truth = train.dat$y)
#2 misclassifications


#Cost = 1
svmfit.train4 <- svm(y ~ ., data = train.dat, kernel = "linear", cost=1)
pred.train4 <- predict(svmfit.train4, train.dat)
table(predict = pred.train4, truth = train.dat$y)
#2 misclassifications


#Cost = 10
svmfit.train5 <- svm(y ~ ., data = train.dat, kernel = "linear", cost=10)
pred.train5 <- predict(svmfit.train5, train.dat)
table(predict = pred.train5, truth = train.dat$y)
#0 misclassifications


#Cost = 100
svmfit.train6 <- svm(y ~ ., data = train.dat, kernel = "linear", cost=100)
pred.train6 <- predict(svmfit.train6, train.dat)
table(predict = pred.train6, truth = train.dat$y)
#0 misclassifications


#Cost = 1000
svmfit.train7 <- svm(y ~ ., data = train.dat, kernel = "linear", cost=1000)
pred.train7 <- predict(svmfit.train7, train.dat)
table(predict = pred.train7, truth = train.dat$y)
#0 misclassifications


#Cost = 10000
svmfit.train8 <- svm(y ~ ., data = train.dat, kernel = "linear", cost=10000)
pred.train8 <- predict(svmfit.train8, train.dat)
table(predict = pred.train8, truth = train.dat$y)
#0 misclassifications


##These misclassifications relate relatively similarly to the cross-validation
##error rates produced by the tune model. Costs of 100, 1000, and 10000 produce
##error rates of 0.00 when cross-validated, so it makes sense that they would have
##zero misclassifications. Costs 0.001, 0.01, 0.1, and 1 all has some misclassifications
##that generally follow the relative amount of cross-validation error associated
##with each cost value. What is undetermind is why the cost value of 10 has zero
##misclassifications while having a non-zero cross-validation error rate.
##This could potentially have to do with the sample size/range of data that we
##arbitrarily chose for this problem, or some other factor.


###################################################################


##6C
#Generate an appropriate test data set, and compute the test
#errors corresponding to each of the values of cost considered.
#Which value of cost leads to the fewest test errors, and how
#does this compare to the values of cost that yield the fewest
#training errors and the fewest cross-validation errors?


##Generating test data set
xtest <- matrix(rnorm(140 * 2), ncol = 2)
ytest <- sample(c(-1, 1), 140, rep = TRUE)
xtest[ytest == 1, ] <- xtest[ytest == 1, ] + 3
test.dat <- data.frame(x = xtest, y = as.factor(ytest))


#Determining how many points were miss classified for each cost value using test data
#Cost = 0.001
pred.test1 <- predict(svmfit.train1, test.dat)
table(predict = pred.test1, truth = test.dat$y)
#4 misclassifications


#Cost = 0.01
pred.test2 <- predict(svmfit.train2, test.dat)
table(predict = pred.test2, truth = test.dat$y)
#5 misclassifications


#Cost = 0.1
pred.test3 <- predict(svmfit.train3, test.dat)
table(predict = pred.test3, truth = test.dat$y)
#6 misclassifications


#Cost = 1
pred.test4 <- predict(svmfit.train4, test.dat)
table(predict = pred.test4, truth = test.dat$y)
#5 misclassifications


#Cost = 10
pred.test5 <- predict(svmfit.train5, test.dat)
table(predict = pred.test5, truth = test.dat$y)
#6 misclassifications


#Cost = 100
pred.test6 <- predict(svmfit.train6, test.dat)
table(predict = pred.test6, truth = test.dat$y)
#6 misclassifications


#Cost = 1000
pred.test7 <- predict(svmfit.train7, test.dat)
table(predict = pred.test7, truth = test.dat$y)
#6 misclassifications


#Cost = 10000
pred.test8 <- predict(svmfit.train8, test.dat)
table(predict = pred.test8, truth = test.dat$y)
#6 misclassifications


##What we find here is that the lowest cost value that has the fewest misclassifications
##is 0.001, our minimal value. Not only that, in general the higher numbers of
##misclassification occurred out our higher cost values. This is what the initial
##statement that motivated this problem predicted, that a smaller cost value would
##perform better on test data than a larger one. We see this when we compare the
##test error misclassification to both the train error misclassification and
##cross-validation training error.



######################################################################



##6D
#Discuss your results.



##This is the classic case of bias/variance trade-off in machine learning.
##When creating a model, one must balance the line between pushing for maximum
##performance and over fitting. In the case of our training results, our model
##was able to achieve zero error and zero misclassifications because it was able
##to perfectly adjust itself, squeezing in its margins until a perfect 1-D
##hyper plane was achieved. When compared to the test data, these models involving
##high cost parameters performed poorly due to their over fit, making them unrealistic
##for real data. Smaller cost values perform better when compared to test data
##due to there wide margins and increased flexibility.


#cost = 0.01 summary index's and plot
svmfit.train2$index
summary(svmfit.train2)
plot(svmfit.train2, train.dat)


#cost = 100 summary index's and plot
svmfit.train6$index
summary(svmfit.train6)
plot(svmfit.train6, train.dat)

