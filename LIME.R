#load required packages

library(lime)       # local interpretation
library(ggplot2)    #  visualization 
library(caret)      #  model building
library(DMwR)       # SMOTE

#--------- Data checking
load("/Users/..../Churn.RData")
sum(is.na(data)) #no missing values
str(data) 

#--------- Split training and test dataset
set.seed(1998)
dt = sort(sample(nrow(data), nrow(data)*.999))
train_obs<-data[dt,]
test_obs<-data[-dt,]

#------- SMOTE for unbalanced dataset.
new_train_obs<-SMOTE(Churn~.,train_obs, perc.over = 100,perc.under = 300)
prop.table(table(new_train_obs$Churn)) #Yes3730 #No5598
prop.table(table(train_obs$Churn)) #1866 #5161

#------- randomeforest (compare with more balanced training set)
fit.caret <- caret::train(Churn ~ ., data = train_obs, 
                          method = "ranger",
                          trControl = trainControl(method = "cv", number = 5, classProbs = TRUE),
                          tuneLength = 1,
                          importance="impurity")

fit.caret2 <- caret::train(Churn ~ ., data = new_train_obs, 
                           method = "ranger",
                           trControl = trainControl(method = "cv", number = 5, classProbs = TRUE),
                           tuneLength = 1,
                           importance="impurity") #better Accuracy and Kappa 

#predict with the test set with 8 observations (same result)
predict<-predict(fit.caret2,test_obs)
confusionMatrix(predict,test_obs$Churn) 
confusionMatrix(predict(fit.caret,test_obs),test_obs$Churn) 

# ---------- use LIME to interpret the result
#create an explainer object
explainer_caret <- lime(new_train_obs, fit.caret2, n_bins = 5)

class(explainer_caret)
summary(explainer_caret)

#explain new observation 
#case 3 and 1 in the test set

explanation_caret <- lime::explain(
  x = test_obs[,-13], 
  explainer = explainer_caret, 
  n_permutations = 500,
  dist_fun = "gower",
  kernel_width = .75,
  n_features = 10, 
  feature_select = "lasso_path",
  n_labels = 1
)

#tuned versions
explanation_caret_tuned <- lime::explain(
  x = test_obs[,-13], 
  explainer = explainer_caret, 
  n_permutations = 500,
  dist_fun = "manhattan",
  kernel_width = 3,
  n_features = 10, 
  feature_select = "lasso_path",
  n_labels = 1
)

explanation_caret_1<-lime::explain(
  x = test_obs[1,-13], 
  explainer = explainer_caret, 
  n_permutations = 500,
  dist_fun = "manhattan",
  kernel_width = 3,
  n_features = 10, 
  feature_select = "lasso_path",
  n_labels = 1
)

explanation_caret_4<-lime::explain(
  x = test_obs[4,-13], 
  explainer = explainer_caret, 
  n_permutations = 500,
  dist_fun = "manhattan",
  kernel_width = 3,
  n_features = 10, 
  feature_select = "lasso_path",
  n_labels = 1
)
plot_features(explanation_caret_4) #predict right
plot_features(explanation_caret_1) #predict wrong

