palette(c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
          "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"))

library(shiny)
library(shinyWidgets)
library(lattice)
library(ggplot2)
library(RColorBrewer)
library(tidyverse)
library(rlang)
library(plyr)# for mean calculation
source("decisionplot.R")
library(caret) # knn
library(naivebayes)
library(e1071) # svm
library(nnet) # nn
library(randomForest)
library(MASS) # lda

# minimal ggplot theme 
theme_set(
  theme_minimal() +
    theme(legend.position = "right")
)

ui = fluidPage(
  tags$head(
    tags$link(rel="stylesheet", 
              href="https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/katex.min.css", 
              integrity="sha384-dbVIfZGuN1Yq7/1Ocstc1lUEm+AT+/rCkibIcC/OmWo5f0EA48Vf8CytHzGrSwbQ",
              crossorigin="anonymous"),
    HTML('<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/katex.min.js" integrity="sha384-2BKqo+exmr9su6dir+qCw08N2ZKRucY4PrGQPPWU1A7FtlCGjmEGFqXCv5nyM5Ij" crossorigin="anonymous"></script>'),
    HTML('<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous"></script>'),
    HTML('
    <script>
      document.addEventListener("DOMContentLoaded", function(){
        renderMathInElement(document.body, {
          delimiters: [{left: "$", right: "$", display: false}]
        });
      })
    </script>')
  ),
  titlePanel("Interactive Visualization of Popular Supervised Learning Methods with Iris", windowTitle = "My First Shiny App"),
  navlistPanel(widths = c(3,9),
               "The Data",
               tabPanel("Descriptive Plots (1)",
                        fluidRow(
                          column(6, tags$h4("Scatterplot of two selected predictors with fitted linear regression lines and 95% confidence bands."),
                                 wellPanel(selectInput('xcol0', 'X Variable', names(iris)[1:4], selected = names(iris)[[1]]),
                                           selectInput('ycol0', 'Y Variable', names(iris)[1:4], selected = names(iris)[[2]])),
                                 plotOutput("Scatterplot")),
                          column(6, tags$h4("Boxplot of selected predictor distinguished by class: Species."),
                                 wellPanel(selectInput('ybox', 'Y Variable', names(iris)[1:4], selected = names(iris)[[2]])),
                                 plotOutput('Boxplot')))),
               
               tabPanel("Descriptive Plots (2)",
                        fluidRow(
                          column(12, tags$h4("Density plot of selected predictor distinguished by class (Species) with plotted mean values."),
                                 wellPanel(selectInput('xdens', 'X Variable', names(iris)[1:4], selected= names(iris)[2])),
                                 plotOutput("Densplot")))
               ),
               "Classification",
               
               tabPanel("K-Nearest Neighbors",
                        helpText("The K-Nearest Neighbors classifier works by identifying the K observations in the training
                        data nearest to a given observation $x_0$, $\\textit{N}$. It will then estimate the $\\textbf{conditional probability}$
                        for class $\\textit{j}$ as the fraction of points whose response values equal $\\textit{j}$.
                        KNN uses the following conditional probability equation:",
                                 tags$br(), 
                                 tags$br(),
                                 "$Pr(Y=j|X=x_0)=\\frac{1}{K}$$\\sum_{i \\in N}$$I(y_i=j)$",
                                 tags$br(),
                                 tags$br(),
                                 "KNN will assign the test observation $x_0$ to the class with the largest probability."),
                        wellPanel(selectInput('xcol', 'X Variable', names(iris)[1:4], selected = names(iris)[[1]]),
                                  selectInput('ycol', 'Y Variable', names(iris)[1:4], selected = names(iris)[[2]]),
                                  sliderInput('K', 'Size K', 3, min = 1, max = 25)),
                        plotOutput('KNNplot'), "Decision plot format credited to Michael Hahsler"),
               
               tabPanel("Naive Bayes Classifier",
                        helpText("The Naive Bayes classifier is based on $\\textbf{Bayes' Theorem}$:",
                                 tags$br(),
                                 tags$br(),
                                 "$P(B|A)=\\frac{P(B)P(A|B)}{P(A)}$",
                                 tags$br(),
                                 tags$br(),
                                 "With the goal being to evaluate an observation $x_0$, with $p$ associated predictors $(X_1=x_1,X_2=x_2,...,X_p=x_p)$,
                                 as belonging to one of $K$ classes we can generate the conditional probability for $x_0$ belonging to
                                 some class C based on its predictor values: $P(x_0=C_k|x_1,x_2,...,x_p)$",
                                 tags$br(),
                                 "From this, we can see how this easily fits into Bayes' Theorem:",
                                 tags$br(),
                                 tags$br(),
                                 "$P(x_0=C_k|x_1,x_2,...,x_p)=\\frac{P(C_k)P(x_1,x_2,...,x_p|C_k)}{P(x_1,x_2,...,x_p)}$",
                                 tags$br(),
                                 tags$br(),
                                 "Naive Bayes operates under the $\\textbf{conditional independence assumption}$ which means that for each class
                                 $C_k$ the values of the predictors are independent of each other. This means that the numerator from the previous equation
                                 , which is the joint probability of class $C_k$ and the data ($X_1=x_1,X_2=x_2,...,X_p=x_p$), can be broken down into a 
                                 product of probabilities:",
                                 tags$br(),
                                 tags$br(),
                                 "$P(C_k \\cap \\{x_1,x_2,...,x_p\\})=P(C_k,x_1,x_2,...,x_p)=P(C_k)\\Pi_{i=1}^{p}P(x_i|C_k)$",
                                 tags$br(),
                                 tags$br(),
                                 " Thus we have:",
                                 tags$br(),
                                 tags$br(),
                                 "$P(C_k|x_1,x_2,...,x_p)=\\frac{P(C_k)\\Pi_{i=1}^{p}P(x_i|C_k)}{P(x_1,x_2,...,x_p)}$",
                                 tags$br(),
                                 tags$br(),
                                 "This equation has roots in Bayesian estimation, where $P(C_k)$ is the prior probability and $\\Pi_{i=1}^{p}P(x_i|C_k)$
                                 represents the likelihood function that is based on conditional probability distributions. The denominator is simply a 
                                 constant. Thus, assigning a class to observation $x_0$ is accomplished by evaluating the numerator for each class $K$
                                 and determining which class gives the highest value."),
                        wellPanel(selectInput('xcol5', 'X Variable', names(iris)[1:4], selected = names(iris)[[1]]),
                                  selectInput('ycol5', 'Y Variable', names(iris)[1:4], selected = names(iris)[[2]])),
                        plotOutput('NBplot'),"Decision plot format credited to Michael Hahsler"),
               
               tabPanel("Linear Discriminant Analysis",
                        helpText("Linear Discriminant Analysis for $p > 1$ predictors follows the assumption that the observations
                                 for the $k^{th}$ class are $\\textbf{multivariate normally distributed}$, with individual mean vectors $\\mu_k$, but
                                 covariance matrix $\\Sigma$ common to all $\\textit{K}$ classes. By plugging in the multivariate 
                                 normal density function for the $k^{th}$ class into the equation that is Bayes' Theorem, the resultant
                                 Bayes classifier assigns observation $x_0$ to the class that maximizes the vector/matrix equation:",
                                 tags$br(), 
                                 tags$br(),
                                 "$\\delta_k(x)=x^T\\Sigma^{-1}\\mu_k$$-\\frac{1}{2}\\mu_{k}^{T}\\Sigma^{-1}\\mu_k$$+log(\\pi_k)$",
                                 tags$br(),
                                 tags$br(),
                                 "Note that $\\pi_k$ represents the prior probability of a chosen observation belonging to the $k^{th}$ class."),
                        wellPanel(selectInput('xcol6', 'X Variable', names(iris)[1:4], selected = names(iris)[[1]]),
                                  selectInput('ycol6', 'Y Variable', names(iris)[1:4], selected = names(iris)[[2]])),
                        plotOutput('LDAplot'), "Decision plot format credited to Michael Hahsler"),
               
               tabPanel("Support Vector Machine (C-classification)",
                        helpText("The support vector machine is a great means for classifying data that has a non-linear boundary. It
                                 works by enlarging the feature space by means of kernel functions. The support vector machine is related
                                 to the maximal margin classifier by extension of the $\\textbf{support vector classifier}$. 
                                 To avoid going into too much technical depth for this app, concepts will be provided in brief.",
                                 tags$br(),
                                 tags$br(),
                                 "A $\\textbf{hyperplane}$ (in $p$-dimensional space it is a subspace of $(p-1)$-dimension) can be used for 
                                 a binary classification problem, where any given point either lies on or to either side of the hyperplane.
                                 This hyperplane seeks to separate observations by class.",
                                 tags$br(),
                                 tags$br(),
                                 "There are infinitely many hyperplanes. The maximal margin hyperplane, obtained from the 
                                 $\\textbf{maximal margin classifier}$ is the optimal hyperplane that has the farthest
                                 minimal distance (defined by a margin) from all data observations. Points that lie on the margin 
                                 are $\\textbf{support vectors}$. This optimal hyperplane is intended to boost the accuracy
                                 in predicting test observations. However, this hyperplane is susceptible to overfitting when $p$ is large.",
                                 tags$br(),
                                 tags$br(),
                                 "$\\textbf{Support vector classifiers}$ attempt to reduce overfitting by generating a soft margin that allows 
                                 for some points to be within the margin and even on either side of the hyperplane. When the support vector 
                                 classifier has a non-linear kernel, the classifier is termed a $\\textbf{support vector machine}$.",
                                 tags$br(),
                                 tags$br(),
                                 "A $\\textbf{kernel}$ written as $K(x,y)$, where $x$ and $y$ are vectors, is used to quantify the similarity between two observations by means of 
                                 taking the inner product of the observations.",
                                 tags$br(),
                                 tags$br(),
                                 "A $\\textbf{linear}$ kernel quantifies the similarity of two observations using their standard correlation. It has an optional
                                 adjustable constant $c$:",
                                 tags$br(),
                                 "$K(x,y)=x^Ty+c$",
                                 tags$br(),
                                 tags$br(),
                                 "A $\\textbf{polynomial}$ kernel fits the classifier in a dimensional space higher than that of the original $p$D space. It has an 
                                 adjustable slope $a$ and constant $c$:",
                                 tags$br(),
                                 "$K(x,y)=(ax^Ty+c)^d$",
                                 tags$br(),
                                 tags$br(),
                                 "The $\\textbf{radial basis function (RBF)}$ kernel fits the classifier based on Euclidean distance. Training observations that are farther from
                                 an observation $x_0$ generate a large inner product that corresponds to a small value when exponentiated in the equation. This
                                 means that observations farther away will have little effect on classifying $x_0$. This leads to a localizing effect
                                 on the classifier, where only close observations to $x_0$ have influence on classification. This classifier uses an adjustable parameter $\\sigma$:",
                                 tags$br(),
                                 "$K(x,y)=exp(-\\frac{||x-y||^2}{2\\sigma^2})$",
                                 tags$br(),
                                 tags$br(),
                                 "A $\\textbf{sigmoid}$ kernel is similar to a two-layer perceptron neural network:",
                                 tags$br(),
                                 "$K(x,y)=tanh(ax^Ty+c)$",
                                 tags$br(),
                                 tags$br(),
                                 "**For all kernels, default parameters were used. Information on this can be found in", 
                                 tags$code('?svm()'),
                                 ".**"
                        ),
                        wellPanel(selectInput('xcol2', 'X Variable', names(iris)[1:4], selected = names(iris)[[1]]),
                                  selectInput('ycol2', 'Y Variable', names(iris)[1:4], selected = names(iris)[[2]]),
                                  radioGroupButtons(inputId = 'rad1', label='Type', choices=c('linear','polynomial','radial','sigmoid'),
                                                    status='primary', selected = 'linear')),
                        plotOutput('SVMplot')),
               
               tabPanel("Single-Layer Neural Network",
                        helpText("Computational neural networks begin with the $\\textbf{perceptron}$. A perceptron is an artificial neuron
                                 that acts like a binary classifier that consists of a layer of inputs multiplied by their associated weights (or bias). 
                                 These products added together form the weighted sum. A step function known as the $\\textbf{activation function}$
                                 is applied to the sum. The output layer has a threshold, similar to a biological neuron, and when the threshold is met
                                 based on the value of the weighted sum, the neuron is activated (eg. output=1).", 
                                 tags$br(),
                                 tags$br(),
                                 "For the purpose of this application, a feedforward single layer neural network was applied. Single layer neural networks
                                 consist of one hidden layer of neurons, with number of neurons in the layer $n$. In R, if the output is a factor
                                 with level > 2, the number of outputs in the output layer equals the number of levels, and classification
                                 follows the maximum conditional likelihood fitting.",
                                 tags$br(),
                                 tags$br(),
                                 "Optimization is normally included in neural network learning. A $\\textbf{cost function}$ $J(\\Theta)$ is constructed which compares the outputs
                                 of a neural network to the known classifications of the training data. The optimizer adjusts the weights of the
                                 neural network connections in order to minimize the value of the cost function to provide the
                                 best classifications of the data. A general equation for the cost function is included below:",
                                 tags$br(),
                                 tags$br(),
                                 "$J(\\Theta)=-\\frac{1}{m}[\\sum_{i=1}^{m}\\sum_{k=1}^{K}y_k^ilog(h_{\\Theta}(x^i))_k+(1-y_k^i)log(1-(h_{\\Theta}(x^i))_k)]$",
                                 tags$br(),
                                 tags$br(),
                                 "$m$ represents training data observations, $K$ represents the class, $\\Theta$ represents the matrix of weights associated with 
                                 the neurons, and $h_{\\Theta}(x^i))$ represents the predictions of the neural network for a given $x$."
                        ),
                        wellPanel(selectInput('xcol3', 'X Variable', names(iris)[1:4], selected = names(iris)[[1]]),
                                  selectInput('ycol3', 'Y Variable', names(iris)[1:4], selected = names(iris)[[2]]),
                                  sliderTextInput(inputId = 'slide1', label='Choose value', choices=c(1, seq(,150, 5, length.out=30)),
                                                  grid=T, selected = 1)),
                        plotOutput('NNplot'), "Decision plot format credited to Michael Hahsler"),
               
               tabPanel("Random Forest",
                        helpText("Before talking about Random Forests, there are a couple things to mention first. A Random Forest 
                                 is a Tree-based classification method that generates predictions
                                 based on the most commonly occuring class in a given region. A classification tree is generated
                                 by recursive binary splitting that is determined by an error rate, such as the $\\textbf{Gini Index}$:",
                                 tags$br(),
                                 tags$br(),
                                 "$G=\\sum_{k=1}^K\\hat{p}_{mk}(1-\\hat{p}_{mk})$",
                                 tags$br(),
                                 tags$br(),
                                 "Here $\\hat{p}_{mk}$ stands for the proportion of observations in the $m^{th}$ region of the $k^{th}$ class. 
                                 The Gini Index is also called the Gini Impurity Index. With a range from 0 to 1, an Index value of 0 denotes that the node 
                                 contains observations of a single class, and 1 signifies that the node contains observations randomly 
                                 distributed across multiple classes.",
                                 tags$br(),
                                 "Decision trees often yield highly variable results. 
                                 By implementing bootstrap resampling methods and taking the average of generated sample
                                 predictions, the variance of this statistical learning method can be reduced. This method is called 
                                 $\\textit{bagging}$.",
                                 tags$br(), 
                                 tags$br(),
                                 "A Random Forest improves upon bagging by decorrelating the trees. It does so by reselecting an
                                 $\\textit{m}$ sized subset of predictors at each split in the tree, where $\\textit{m}=\\sqrt{p}$ usually. This causes
                                 the probability for a strong predictor in the dataset to be considered, on average, $\\frac{(p-m)}{p}$
                                 of the time. This decorrelates the predictions from the trees, gives other predictors a chance, and 
                                 ultimately reduces variability.",
                                 tags$br(),
                                 tags$br(), 
                                 "**To allow for 2-D representation, where $p=2$, $\\textit{m}=1$**",
                                 tags$br(),
                                 "**Default parameters were used. More information can be found in",
                                 tags$code('?randomForest()'),
                                 ".**"
                        ),
                        wellPanel(selectInput('xcol4', 'X Variable', names(iris)[1:4], selected = names(iris)[[1]]),
                                  selectInput('ycol4', 'Y Variable', names(iris)[1:4], selected = names(iris)[[2]])),
                        plotOutput('RFplot'), "Decision plot format credited to Michael Hahsler")
  ))
