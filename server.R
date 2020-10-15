server = function(input, output) {
  
  ##############################
  ##  Descriptive Statistics  ##
  ##############################
  
  # Scatterplot
  q = reactive({ggplot(iris, aes_(x=iris[[as.name(input$xcol0)]], y=iris[[as.name(input$ycol0)]]))})
  
  sq = reactive({q() + geom_point(aes(color = Species), size=2.2) + theme(legend.position = "top")})
  
  output$Scatterplot = renderPlot({sq() +  geom_smooth(aes(color = Species, fill = Species), method = "lm") +
      scale_color_brewer(palette = "Dark2") + scale_fill_brewer(palette="Dark2") + xlab(input$xcol0) + ylab(input$ycol0)})
  
  
  # Boxplot
  g = reactive({ggplot(data=iris, aes_('Species', y=iris[[as.name(input$ybox)]]))+ theme(legend.position = "top")})
  
  output$Boxplot = renderPlot(g() + geom_boxplot(aes(fill=Species)) + scale_fill_brewer(palette = "Set2") + 
                                ylab(input$ybox) + xlab(" "))
  
  
  # Densityplot
  c = reactive({ggplot(data=iris, aes_(iris[[as.name(input$xdens)]]))})
  
  mu_df = reactive({iris %>%
      group_by(Species) %>%
      summarise(grp.mean=mean(!! rlang::sym(input$xdens))) # set so that can read input and divide the groups
  })
  
  dc = reactive({c() + geom_density(kernel="gaussian", aes(color=Species, fill=Species), alpha=0.4) +
      geom_vline(data=mu_df(), aes(xintercept=grp.mean, color=Species), linetype="dashed") +
      theme(legend.position = "top") + xlab(input$xdens)})
  
  output$Densplot = renderPlot(dc() + scale_color_brewer(palette="Dark2") + scale_fill_brewer(palette = "Dark2"))
  
  ###########
  ##  KNN  ##
  ###########
  
  selection <- reactive({
    df1 <- data.frame(x=iris[[input$xcol]])
    df2 <- data.frame(y=iris[[input$ycol]])
    df3 <- data.frame(Species=iris[, "Species"])
    df <- cbind(df1,df2,df3)
    df})
  
  fit = reactive({knn3(Species ~ ., data=selection(), k = input$K)}) 
  
  output$KNNplot = renderPlot({decisionplot(fit(), selection(), class='Species', 
                                            main=paste('Decision Boundary for KNN (',input$K,')', sep=' '))})
  
  ###########
  ##  SVM  ##
  ###########
  
  selection2 <- reactive({
    df1 <- data.frame(x=iris[[input$xcol2]])
    df2 <- data.frame(y=iris[[input$ycol2]])
    df3 <- data.frame(Species=iris[, "Species"])
    df <- cbind(df1,df2,df3)
    df})
  
  fit2 = reactive({svm(Species ~ ., data=selection2(), kernel=input$rad1)})
  
  output$SVMplot = renderPlot({decisionplot(fit2(), selection2(), class = "Species",
                                            main = paste("Decision Boundary for SVM (",input$rad1,")", sep=' '))})
  
  #################
  ##  NeuralNet  ##
  #################
  
  selection3 <- reactive({
    df1 <- data.frame(x=iris[[input$xcol3]])
    df2 <- data.frame(y=iris[[input$ycol3]])
    df3 <- data.frame(Species=iris[, "Species"])
    df <- cbind(df1,df2,df3)
    df})
  
  fit3 = reactive({nnet(Species ~ ., data=selection3(), size=input$slide1, maxit=1000, trace=F)})
  
  output$NNplot = renderPlot({decisionplot(fit3(), selection3(), class = "Species",
                                           main = paste("Decision Boundary for Neural Net (",input$slide1,")", sep=' '))})
  
  ######################
  ##  Random Forest   ##
  ######################
  
  selection4 <- reactive({
    df1 <- data.frame(x=iris[[input$xcol4]])
    df2 <- data.frame(y=iris[[input$ycol4]])
    df3 <- data.frame(Species=iris[, "Species"])
    df <- cbind(df1,df2,df3)
    df})
  
  fit4 = reactive({randomForest(Species ~ ., data=selection4())})
  
  output$RFplot = renderPlot({decisionplot(fit4(), selection4(), class = "Species",
                                           main = "Decision Boundary for Random Forest")})
  
  ###################
  ##  Naive Bayes  ##
  ###################
  
  selection5 <- reactive({
    df1 <- data.frame(x=iris[[input$xcol5]])
    df2 <- data.frame(y=iris[[input$ycol5]])
    df3 <- data.frame(Species=iris[, "Species"])
    df <- cbind(df1,df2,df3)
    df})
  
  fit5 = reactive({naive_bayes(Species ~ ., data=selection5(), usekernel = T)})
  
  output$NBplot = renderPlot({decisionplot(fit5(), selection5(), class = "Species",
                                           main = "Decision Boundary for Naive Bayes Classifier")})
  
  ####################################
  ##  Linear Discriminant Analysis  ##
  ####################################
  
  selection6 <- reactive({
    df1 <- data.frame(x=iris[[input$xcol6]])
    df2 <- data.frame(y=iris[[input$ycol6]])
    df3 <- data.frame(Species=iris[, "Species"])
    df <- cbind(df1,df2,df3)
    df})
  
  fit6 = reactive({lda(Species ~ ., data=selection6())})
  
  output$LDAplot = renderPlot({decisionplot(fit6(), selection6(), class = "Species",
                                            main = "Decision Boundary for Linear Discriminant Analysis")})
  
}

shinyApp(ui = ui, server = server)
