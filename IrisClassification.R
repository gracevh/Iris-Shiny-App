library(caret)
library(lattice)
library(ggplot2)
source('decisionplot.R')

#Full dataset
y = iris 
# Partial dataset for 2D plotting
x = iris[1:150, c("Sepal.Length", "Sepal.Width", "Species")]

# KNN
model = knn3(Species ~ ., data=x, k = 15)
plot(y$Sepal.Length,y$Petal.Length, col = y[,3])
decisionplot(model, s, class = "Species", main = "kNN (1)")

# Naive Bayes
# Handles conditional densities for class compared to e1071 package
library(naivebayes)
model = naive_bayes(Species~., data=x, usekernel = T)
decisionplot(model, x, class='Species', main = 'naive bayes')

# SVM
model <- svm(Species ~ ., data=x, kernel="linear")
decisionplot(model, x, class = "Species", main = "SVD (linear)")


model <- svm(Species ~ ., data=x, kernel="radial")
decisionplot(model, x, class = "Species", main = "SVD (Radial)")

model <- svm(Species ~ ., data=x, kernel="polynomial")
decisionplot(model, x, class = "Species", main = "SVD (Polynomial)")

model <- svm(Species ~ ., data=x, kernel="sigmoid")
decisionplot(model, x, class = "Species", main = "SVD (Sigmoid)")

# Neural Net
library(nnet)
model <- nnet(Species ~ ., data=x, size = 1, maxit = 1000, trace = FALSE)
decisionplot(model, x, class = "Species", main = "NN (1)")

model <- nnet(Species ~ ., data=x, size = 10, maxit = 1000, trace = FALSE)
decisionplot(model, x, class = "Species", main = "NN (10)")

model <- nnet(Species ~ ., data=x, size = 100, maxit = 1000, trace = FALSE)
decisionplot(model, x, class = "Species", main = "NN (100)")


# Random Forest
library(randomForest)
model <- randomForest(Species ~ ., data=x)
decisionplot(model, x, class = "Species", main = "Random Forest")


