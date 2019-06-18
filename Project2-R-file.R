
library(tidyverse)
library(caret)
library(MASS)
library(parallel)
library(doParallel)
library(pROC)
library(rpart)
library(grid)

#--------------------------------------- Descriptive statistics ---------------------------------------
marketing.full <- ElemStatLearn::marketing
marketing.full$High <- ifelse(marketing.full$Income>=8,1,0)

stats.summary <- function(df){
  temp <- t(apply(df, 2, mean, na.rm = T))
  temp <- rbind(temp,t(apply(df, 2, median, na.rm = T)))
  temp <- rbind(temp,t(apply(df, 2, min, na.rm = T)))
  temp <- rbind(temp,t(apply(df, 2, max, na.rm = T)))
  temp <- rbind(temp,apply(df,2, function(y) sum(length(which(is.na(y))))))
  temp <- round(temp, 3)
  rownames(temp) <- c("Mean","Median","Min","Max","No.NA")
  temp
}

descriptive <- stats.summary(marketing.full)


#--------------------------------------- Plot variables ---------------------------------------
##### Histogram of all variables #####
plot.fn <- function(df){
  t <- ncol(df)
  local({
    for(i in 1:t){
      df[,names(df)] <- lapply(df[,names(df)] , factor)
      assign(paste0("p",i), eval(parse(text = paste0("qplot(df[,",i,"],data = df, xlab = \"",
                                                     colnames(df)[i], "\")+theme_minimal()"))))
    }
    mylist <- mget(ls(pattern = "p."))
    gridExtra::grid.arrange(grobs = mylist,nrow = 3) 
  })
}
plot.fn(marketing.full)

##### Plot all variables as a function of High #####
dist.fn <- function(df, nrow = 3){
  t <- ncol(df)
  local({
    for(i in 2:(t)){
      df[,names(df)] <- lapply(df[,names(df)] , factor)
      temp <- data.frame(table(df[,"High"],df[,i]))
      names(temp) <- c("High",colnames(df)[i],"Count")
      assign(paste0("p",i), eval(parse(text =paste0("ggplot(data = temp, aes(x = High, y = Count, fill = ",
                                                    colnames(df)[i], "))+ geom_bar(stat=\"identity\")+
                                                    theme_minimal()"))))
    }
    mylist <- mget(ls(pattern = "p."))
    gridExtra::grid.arrange(grobs = mylist,nrow = nrow) 
  })
}
marketing.graph <- subset(marketing.mean.lived, select = c(High,Edu,Status,Marital, Occupation))
dist.fn(marketing.graph, nrow = 2)



gridExtra::grid.arrange(ggplot())

#--------------------------------------- Data cleaning ---------------------------------------
create.data <- function(){
  ##### Remove Income #####
  marketing.full <- ElemStatLearn::marketing
  marketing.full$High <- ifelse(marketing.full$Income>=8,1,0)
  marketing.full$High <- as.factor(marketing.full$High)
  levels(marketing.full$High) <- c("zero","one")
  
  marketing <<- subset(marketing.full, select = -c(Income, Lived))
  marketing.lived <<- subset(marketing.full, select = -c(Income))
  
  
  ##### Mean imputation #####
  impute.mean <- function(df){
    for(i in 1:ncol(df)){
      for(j in 1:nrow(df)){
        if(is.na(df[j,i])){
          df[j,i] <- as.integer(mean(df[,i], na.rm = T))
        }
      }
    }
    df
  }
  
  marketing.mean <<- impute.mean(marketing)
  marketing.mean.lived <<- impute.mean(marketing.lived)
  #sum(is.na(marketing.mean))
  
  
  ##### Remove NAs #####
  marketing <<- na.omit(marketing)
  marketing.lived <<- na.omit(marketing.lived)
  
  mark <<- list(marketing.mean.lived=marketing.mean.lived,marketing = marketing,
                marketing.mean = marketing.mean, marketing.lived=marketing.lived)
}


#--------------------------------------- Models ---------------------------------------
envir.clean <- c("envir.clean","create.data")
rm(list=setdiff(ls(), envir.clean))
create.data()
seed <- 14
percent <- 0.75
fit_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                            summaryFunction=twoClassSummary)
#df <- marketing.mean.lived
rm(list="nogo")
cl <- makeCluster(detectCores())
doParallel::registerDoParallel(cl)

for(i in 1:length(mark)){
  nogo <- "nogo"
  df <- mark[[i]]
  
  
  ############# LDA #############
  
  lda.fn <- function(df){
    
    if(!exists("nogo")){
      cl <- makeCluster(detectCores())
      doParallel::registerDoParallel(cl)
    }
    
    
    set.seed(seed)
    n <- nrow(df)
    shuffled <- df[sample(n),]
    train.lda <- shuffled[1:round(percent * n),]
    test.lda <<- shuffled[(round(percent * n) + 1):n,]
    rm(list="shuffled")
    
    mod.lda <<- train(High ~ ., data=train.lda, method="lda",
                      trControl = fit_control,
                      metric = "ROC")
    
    pred.lda <<- predict(mod.lda, newdata = test.lda, type = "prob")
    test.lda$pred <<- ifelse(pred.lda$zero>0.5, "zero","one")
    
    if(!exists("nogo")){
      stopCluster(cl)
      registerDoSEQ()
    }
  }
  
  lda.fn(df)
  
  
  #table(test.lda$pred, test.lda$High)
  #caret::confusionMatrix(test.lda$High,test.lda$pred)
  
  accuracy.lda = round(mean(test.lda$pred == test.lda$High)*100,2)
  
  
  #print(lda.mod)
  
  # Plot LDA
  ROC.lda <- roc(as.numeric(test.lda$High),as.numeric(pred.lda$one))
  #plot(ROC.lda, col = "red")
  #auc(ROC.lda)
  
  
  ############# GBM #############
  
  gbm.fn <- function(df){
    
    if(!exists("nogo")){
      cl <- makeCluster(detectCores())
      doParallel::registerDoParallel(cl)}
    
    
    set.seed(seed)
    n <- nrow(df)
    shuffled <- df[sample(n),]
    train.gbm <- shuffled[1:round(percent * n),]
    test.gbm <<- shuffled[(round(percent * n) + 1):n,]
    rm(list="shuffled")
    
    gbmGrid <-  expand.grid(interaction.depth = c(1,3,5,7,9),
                            n.trees = (1:50)*20,
                            shrinkage = c(0.1,0.01),
                            n.minobsinnode = 10)
    
    mod.gbm <<- train(High ~ ., data=train.gbm, method="gbm",
                      trControl = fit_control,metric = "ROC", tuneGrid = gbmGrid, verbose = FALSE)
    
    pred.gbm <<- predict(mod.gbm, newdata = test.gbm, 
                         n.trees = mod.gbm$results$n.trees[which.max(mod.gbm$results$ROC)],
                         interaction.depth = mod.gbm$results$interaction.depth[which.max(mod.gbm$results$ROC)],
                         shrinkage = mod.gbm$results$shrinkage[which.max(mod.gbm$results$ROC)],
                         type = "prob")
    test.gbm$pred <<- ifelse(pred.gbm$zero>0.5, "zero","one")
    
    if(!exists("nogo")){
      stopCluster(cl)
      registerDoSEQ()
    }
  }
  
  gbm.fn(df)
  
  #table(test.gbm$pred, test.gbm$High)
  #caret::confusionMatrix(test.gbm$High,test.gbm$pred)
  
  accuracy.gbm = round(mean(test.gbm$pred == test.gbm$High)*100,2)
  
  # Plot GBM
  ggplot(mod.gbm)+theme_minimal()
  
  ROC.gbm <- roc(as.numeric(test.gbm$High),as.numeric(pred.gbm$one))
  #plot(ROC.gbm, col = "red")
  auc(ROC.gbm)
  
  #rm(list=setdiff(ls(), envir.clean))
  
  
  
  ############# Logreg #############
  
  log.fn <- function(df){
    
    if(!exists("nogo")){
      cl <- makeCluster(detectCores())
      doParallel::registerDoParallel(cl)
    }
    
    set.seed(seed)
    n <- nrow(df)
    shuffled <- df[sample(n),]
    train.log <- shuffled[1:round(percent * n),]
    test.log <<- shuffled[(round(percent * n) + 1):n,]
    rm(list="shuffled")
    
    mod.log <<- train(High ~., data = train.log, method = "glm",
                      trControl = fit_control, family = binomial, metric = "ROC")
    
    pred.log <<- predict(mod.log, newdata = test.log, type = "prob")
    test.log$pred <<- ifelse(pred.log$zero>0.5,"zero","one")
    
    if(!exists("nogo")){
      stopCluster(cl)
      registerDoSEQ()
    }
  }
  
  log.fn(df)
  
  #table(test.log$pred, test.log$High)
  #caret::confusionMatrix(test.log$High,test.log$pred)
  
  accuracy.log = round(mean(test.log$pred == test.log$High)*100,2)
  
  ROC.log <- roc(as.numeric(test.log$High),as.numeric(pred.log$one))
  #plot(ROC.log, col = "red")
  #auc(ROC.log)
  
  
  ############# Classification with pruning #############
  
  tree.fn <- function(df){
    if(!exists("nogo")){
      cl <- makeCluster(detectCores())
      doParallel::registerDoParallel(cl)
    }
    
    set.seed(seed)
    n <- nrow(df)
    shuffled <- df[sample(n),]
    train.tree <- shuffled[1:round(percent * n),]
    test.tree <<- shuffled[(round(percent * n) + 1):n,]
    rm(list="shuffled")
    
    mod.tree <<- train(High ~., data = train.tree, method = "rpart",
                       trControl = fit_control, metric = "ROC", tuneLength = 10)
    
    pred.tree <<- predict(mod.tree, newdata = test.tree, type = "prob")
    test.tree$pred <<- ifelse(pred.tree$zero>0.5,"zero","one")
    
    
    if(!exists("nogo")){
      stopCluster(cl)
      registerDoSEQ()
    }
  }
  
  tree.fn(marketing.mean.lived)
  
  
  #table(test.tree$pred, test.tree$High)
  #caret::confusionMatrix(test.tree$High,test.tree$pred)
  
  accuracy.tree = round(mean(test.tree$pred == test.tree$High)*100,2)
  
  # Plot tree
  ROC.tree <- roc(as.numeric(test.tree$High),as.numeric(pred.tree$one))
  #plot(ROC.tree, col = "red")
  #auc(ROC.tree)
  
  ############# Random forest #############
  
  rf.fn <- function(df){
    
    if(!exists("nogo")){
      cl <- makeCluster(detectCores())
      doParallel::registerDoParallel(cl)
    }
    
    set.seed(seed)
    n <- nrow(df)
    shuffled <- df[sample(n),]
    train.rf <- shuffled[1:round(percent * n),]
    test.rf <<- shuffled[(round(percent * n) + 1):n,]
    rm(list="shuffled")
    
    mod.rf <<- train(High ~., data = train.rf, method = "rf",
                     trControl = fit_control, metric = "ROC", tuneLength = 10)
    
    pred.rf <<- predict(mod.rf, newdata = test.rf, type = "prob")
    test.rf$pred <<- ifelse(pred.rf$zero>0.5,"zero","one")
    
    if(!exists("nogo")){
      stopCluster(cl)
      registerDoSEQ()
    }
  }
  
  rf.fn(df)
  
  #table(test.rf$pred, test.rf$High)
  #caret::confusionMatrix(test.rf$High,test.rf$pred)
  
  accuracy.rf = round(mean(test.rf$pred == test.rf$High)*100,2)
  
  ROC.rf <- roc(as.numeric(test.rf$High),as.numeric(pred.rf$one))
  #plot(ROC.rf, col = "red")
  #auc(ROC.rf)
  
  
  ############# Metrics #############
  
  summary.all <- data.frame(rbind(cbind(accuracy.lda,auc(ROC.lda)),
                                  cbind(accuracy.gbm,auc(ROC.gbm)),
                                  cbind(accuracy.log,auc(ROC.log)),
                                  cbind(accuracy.tree,auc(ROC.tree)),
                                  cbind(accuracy.rf,auc(ROC.rf))))
  rownames(summary.all) <- c("LDA","GBM","Logreg","Tree","Random Forest")
  colnames(summary.all) <- c("Accuracy","AUC")
  summary.all$AUC <- round(summary.all$AUC*100,2)
  assign(paste0("summary.",names(mark)[i]) ,summary.all)
  
  if(i==1){
    list.all <<- list(lda = list("pred" = pred.lda,"test" = test.lda,"mod" = mod.lda, "roc" = ROC.lda),
                      gbm = list("pred" = pred.gbm,"test" = test.gbm,"mod" = mod.gbm, "roc" = ROC.gbm),
                      log = list("pred" = pred.log,"test" = test.log,"mod" = mod.log, "roc" = ROC.log),
                      tree = list("pred" = pred.tree,"test" = test.tree,"mod" =mod.tree, "roc" = ROC.tree),
                      rf = list("pred" = pred.rf,"test" = test.rf,"mod" = mod.rf, "roc" = ROC.rf)) 
  }
  
  print(paste0(i/length(mark)*100,"%"))
}

stopCluster(cl)
registerDoSEQ()
