# Databricks notebook source
# MAGIC %md
# MAGIC # Training a GLM model with Keras and R
# MAGIC 
# MAGIC This example is a simplified take on training a GLM model using Keras and R. We use MLflow to help us track and log the experiment run. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Installing Dependencies

# COMMAND ----------

install.packages("keras")
install.packages("locfit")
install.packages("corrplot")

# COMMAND ----------

## -----------------------------------------------------------------------------
library(keras)
library(locfit)
library(magrittr)
library(dplyr)
library(tibble)
library(purrr)
library(ggplot2)
library(gridExtra)
library(tidyr)
library(corrplot)
RNGversion("3.5.0")

# COMMAND ----------

install.packages("mlflow")
library(mlflow)
install_mlflow()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading libraries: Tensorflow and Keras
# MAGIC 
# MAGIC After loading in tensorflow, keras, and mlflow, we set up a few variables needed.

# COMMAND ----------

## -----------------------------------------------------------------------------
options(encoding = 'UTF-8')


## -----------------------------------------------------------------------------
# set seed to obtain best reproducibility. note that the underlying architecture may affect results nonetheless, so full reproducibility cannot be guaranteed across different platforms.
seed <- 100
Sys.setenv(PYTHONHASHSEED = seed)
set.seed(seed)
reticulate::py_set_seed(seed)
tensorflow::tf$random$set_seed(seed)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data processing
# MAGIC 
# MAGIC The following cells show how to load data, in this case containing Worker's compensation data. We run some essential data preprocessing for model training.

# COMMAND ----------

## -----------------------------------------------------------------------------
# https://stackoverflow.com/questions/65366442/cannot-convert-a-symbolic-keras-input-output-to-a-numpy-array-typeerror-when-usi
# https://tensorflow.rstudio.com/guide/tfhub/examples/feature_column/
tensorflow::tf$compat$v1$disable_eager_execution()


## -----------------------------------------------------------------------------
ax_limit <- c(0,50000)
line_size <- 1.1


## -----------------------------------------------------------------------------
# MinMax scaler
preprocess_minmax <- function(varData) {
  X <- as.numeric(varData)
  2 * (X - min(X)) / (max(X) - min(X)) - 1
}


## -----------------------------------------------------------------------------
# One Hot encoding for categorical features
preprocess_cat_onehot <- function(data, varName, prefix) {
  varData <- data[[varName]]
  X <- as.integer(varData)
  n0 <- length(unique(X))
  n1 <- 1:n0
  addCols <- purrr::map(n1, function(x, y) {as.integer(y == x)}, y = X) %>%
    rlang::set_names(paste0(prefix, n1))
  cbind(data, addCols)
}


## -----------------------------------------------------------------------------
#https://stat.ethz.ch/pipermail/r-help/2013-July/356936.html
scale_no_attr <- function (x, center = TRUE, scale = TRUE) 
{
    x <- as.matrix(x)
    nc <- ncol(x)
    if (is.logical(center)) {
        if (center) {
            center <- colMeans(x, na.rm = TRUE)
            x <- sweep(x, 2L, center, check.margin = FALSE)
        }
    }
    else if (is.numeric(center) && (length(center) == nc)) 
        x <- sweep(x, 2L, center, check.margin = FALSE)
    else stop("length of 'center' must equal the number of columns of 'x'")
    if (is.logical(scale)) {
        if (scale) {
            f <- function(v) {
                v <- v[!is.na(v)]
                sqrt(sum(v^2)/max(1, length(v) - 1L))
            }
            scale <- apply(x, 2L, f)
            x <- sweep(x, 2L, scale, "/", check.margin = FALSE)
        }
    }
    else if (is.numeric(scale) && length(scale) == nc) 
        x <- sweep(x, 2L, scale, "/", check.margin = FALSE)
    else stop("length of 'scale' must equal the number of columns of 'x'")
    #if (is.numeric(center)) 
    #    attr(x, "scaled:center") <- center
    #if (is.numeric(scale)) 
    #    attr(x, "scaled:scale") <- scale
    x
}

# COMMAND ----------

## -----------------------------------------------------------------------------
square_loss <- function(y_true, y_pred){mean((y_true-y_pred)^2)}
gamma_loss  <- function(y_true, y_pred){2*mean((y_true-y_pred)/y_pred + log(y_pred/y_true))}
ig_loss     <- function(y_true, y_pred){mean((y_true-y_pred)^2/(y_pred^2*y_true))}
p_loss      <- function(y_true, y_pred, p){2*mean(y_true^(2-p)/((1-p)*(2-p))-y_true*y_pred^(1-p)/(1-p)+y_pred^(2-p)/(2-p))}

## k_gamma_loss  <- function(y_true, y_pred){2*k_mean(y_true/y_pred - 1 - log(y_true/y_pred))}
k_ig_loss     <- function(y_true, y_pred){k_mean((y_true-y_pred)^2/(y_pred^2*y_true))}
k_p_loss      <- function(y_true, y_pred){2*k_mean(y_true^(2-p)/((1-p)*(2-p))-y_true*y_pred^(1-p)/(1-p)+y_pred^(2-p)/(2-p))}


## ----------------------------------------------------------------------------- 
## Optional: for plotting and logging purposes

keras_plot_loss_min <- function(x, seed) {
    x <- x[[2]]
    ylim <- range(x)
    vmin <- which.min(x$val_loss)
    df_val <- data.frame(epoch = 1:length(x$loss), train_loss = x$loss, val_loss = x$val_loss)
    df_val <- gather(df_val, variable, loss, -epoch)
    #Added for mlFlow tracking
    plt <- ggplot(df_val, aes(x = epoch, y = loss, group = variable, color = variable)) +
      geom_line(size = line_size) + geom_vline(xintercept = vmin, color = "green", size = line_size) +
      labs(title = paste("Train and validation loss for seed", seed),
           subtitle = paste("Green line: Smallest validation loss for epoch", vmin))
    ggsave("/dbfs/tmp/keras_plot_loss.png")
    suppressMessages(print(plt))
}

# COMMAND ----------

## -----------------------------------------------------------------------------
plot_size <- function(test, xvar, title, model, mdlvariant) {
  out <- test %>% group_by(!!sym(xvar)) %>%
    summarize(obs = mean(Claim) , pred = mean(!!sym(mdlvariant)))
  
  ggplot(out, aes(x = !!sym(xvar), group = 1)) +
    geom_point(aes(y = pred, colour = model)) +
    geom_point(aes(y = obs, colour = "observed")) +
    geom_line(aes(y = pred, colour = model), linetype = "dashed") +
    geom_line(aes(y = obs, colour = "observed"), linetype = "dashed") +
    ylim(ax_limit) + labs(x = xvar, y = "claim size", title = title) +
    theme(legend.position = "bottom")
}

# COMMAND ----------

# DBTITLE 1,Loading in WorkersComp data
## -----------------------------------------------------------------------------
load(file.path("WorkersComp.RData"))  # relative path to .Rmd file

# COMMAND ----------

## -----------------------------------------------------------------------------
dat <- WorkersComp %>% filter(AccYear > 1987, HoursWorkedPerWeek > 0)


## -----------------------------------------------------------------------------
# Order claims in decreasing order for split train/test (see below), and add an ID
dat <- dat %>% arrange(desc(Claim))
dat <- dat %>% mutate(Id=1:nrow(dat))


## -----------------------------------------------------------------------------
# scaling and cut-off
dat <- dat %>% mutate(
        Age = pmax(16, pmin(70, Age)),
        AgeNN = scale_no_attr(Age),
        GenderNN = as.integer(Gender),
        GenderNN = scale_no_attr(GenderNN),
        DependentChildren = pmin(1, DependentChildren),
        DependentChildrenNN = scale_no_attr(DependentChildren),
        DependentsOther = pmin(1, DependentsOther),
        DependentsOtherNN = scale_no_attr(DependentsOther),
        WeeklyPay = pmin(1200, WeeklyPay),
        WeeklyPayNN = scale_no_attr(WeeklyPay),
        PartTimeFullTimeNN = scale_no_attr(as.integer(PartTimeFullTime)),
        HoursWorkedPerWeek = pmin(60, HoursWorkedPerWeek),
        HoursWorkedPerWeekNN = scale_no_attr(HoursWorkedPerWeek),
        DaysWorkedPerWeekNN = scale_no_attr(DaysWorkedPerWeek),
        AccYearNN = scale_no_attr(AccYear),
        AccMonthNN = scale_no_attr(AccMonth),
        AccWeekdayNN = scale_no_attr(AccWeekday),
        AccTimeNN = scale_no_attr(AccTime),
        RepDelay = pmin(100, RepDelay),
        RepDelayNN = scale_no_attr(RepDelay)
)


## -----------------------------------------------------------------------------
# one-hot encoding (not dummy encoding!)
dat <- dat %>% preprocess_cat_onehot("MaritalStatus", "Marital")


## -----------------------------------------------------------------------------
# add two additional randomly generated features (later used)
set.seed(seed)

dat <- dat %>% mutate(
    RandNN = rnorm(nrow(dat)),
    RandNN = scale_no_attr(RandNN),
    RandUN = runif(nrow(dat), min = -sqrt(3), max = sqrt(3)),
    RandUN = scale_no_attr(RandUN)
)

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's take a quick look at the loaded and transformed data

# COMMAND ----------

## -----------------------------------------------------------------------------
head(dat)

# COMMAND ----------

## -----------------------------------------------------------------------------
str(dat)

# COMMAND ----------

## -----------------------------------------------------------------------------
summary(dat)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Finally let's split the data into train and testing datasets

# COMMAND ----------

## -----------------------------------------------------------------------------
idx <- sample(x = c(1:5), size = ceiling(nrow(dat) / 5), replace = TRUE)
idx <- (1:ceiling(nrow(dat) / 5) - 1) * 5 + idx

test <- dat[intersect(idx, 1:nrow(dat)), ]
learn <- dat[setdiff(1:nrow(dat), idx), ]

learn <- learn[sample(1:nrow(learn)), ]
test <- test[sample(1:nrow(test)), ]


## -----------------------------------------------------------------------------
# size of train/test
sprintf("Number of observations (learn): %s", nrow(learn))
sprintf("Number of observations (test): %s", nrow(test))


## -----------------------------------------------------------------------------
# Claims average of learn/test
sprintf("Empirical claims average (learn): %s", round(sum(learn$Claim) / length(learn$Claim), 0))
sprintf("Empirical claims average (test): %s", round(sum(test$Claim) / length(test$Claim), 0))


## -----------------------------------------------------------------------------
# Quantiles of learn/test
probs <- c(.1, .25, .5, .75, .9)
bind_rows(quantile(learn$Claim, probs = probs), quantile(test$Claim, probs = probs))

# COMMAND ----------

## -----------------------------------------------------------------------------
# initialize table to store all model results for comparison
df_cmp <- tibble(
 model = character(),
 learn_p2 = numeric(),
 learn_pp = numeric(),
 learn_p3 = numeric(),
 test_p2 = numeric(),
 test_pp = numeric(),
 test_p3 = numeric(),
 avg_size = numeric(),
)

# COMMAND ----------

## -----------------------------------------------------------------------------
# used/selected features
col_features <- c("AgeNN","GenderNN","DependentChildrenNN","DependentsOtherNN",
                  "WeeklyPayNN","PartTimeFullTimeNN","HoursWorkedPerWeekNN",
                  "DaysWorkedPerWeekNN","AccYearNN","AccMonthNN","AccWeekdayNN",
                  "AccTimeNN","RepDelayNN","Marital1","Marital2","Marital3")
col_names <- c("Age","Gender","DependentChildren","DependentsOther","WeeklyPay",
               "PartTimeFullTime","HoursWorkedPerWeek","DaysWorkedPerWeek",
               "AccYear","AccMonth","AccWeekday","AccTime","RepDelay",
               "Marital1","Marital2","Marital3")


## -----------------------------------------------------------------------------
# select p in [2,3]
p <- 2.5

# COMMAND ----------

## -----------------------------------------------------------------------------
# homogeneous model (learn)
(size_hom <- round(mean(learn$Claim)))
log_size_hom <- log(size_hom)


## -----------------------------------------------------------------------------
df_cmp %<>% bind_rows(
  data.frame(
    model = "Null model",
    learn_p2 = round(gamma_loss(learn$Claim, size_hom), 4),
    learn_pp = round(p_loss(learn$Claim, size_hom, p) * 10, 4),
    learn_p3 = round(ig_loss(learn$Claim, size_hom) * 1000, 4),
    test_p2 = round(gamma_loss(test$Claim, size_hom), 4),
    test_pp = round(p_loss(test$Claim, size_hom, p) * 10, 4),
    test_p3 = round(ig_loss(test$Claim, size_hom) * 1000, 4),
    avg_size = round(size_hom, 0)
  ))
df_cmp

# COMMAND ----------

# MAGIC %md
# MAGIC And now we transform the sliced data into matrices for model processing

# COMMAND ----------

## -----------------------------------------------------------------------------
# Size of input for neural networks
q0 <- length(col_features)
qqq <- c(q0, c(20,15,10), 1)

sprintf("Neural network with K=3 hidden layer")
sprintf("Input feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", qqq[2])
sprintf("Number of hidden neurons second layer: q2 = %s", qqq[3])
sprintf("Number of hidden neurons third layer: q3 = %s", qqq[4])
sprintf("Output dimension: %s", qqq[5])


## -----------------------------------------------------------------------------
# matrices
YY <- as.matrix(as.numeric(learn$Claim))
XX <- as.matrix(learn[, col_features]) 
TT <- as.matrix(test[, col_features])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Architecture
# MAGIC 
# MAGIC Next, we'll use Keras to construct our model architecture! The following is a simplified example of what can be built with Keras and tensorflow. More in depth examples can be found [here](https://blogs.rstudio.com/ai/posts/2018-01-11-keras-customer-churn/).

# COMMAND ----------

## -----------------------------------------------------------------------------
# Size of input for neural networks
q0 <- length(col_features)
qqq <- c(q0, c(20, 15, 10), 1)

sprintf("Neural network with K=3 hidden layer")
sprintf("Input feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", qqq[2])
sprintf("Number of hidden neurons second layer: q2 = %s", qqq[3])
sprintf("Number of hidden neurons third layer: q3 = %s", qqq[4])
sprintf("Number of hidden neurons third layer: q4 = %s", qqq[1])
sprintf("Output dimension: %s", qqq[5])


## -----------------------------------------------------------------------------
# matrices
YY <- as.matrix(as.numeric(learn$Claim))
XX <- as.matrix(learn[, col_features])
TT <- as.matrix(test[, col_features])


## -----------------------------------------------------------------------------
# neural network structure
Design  <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'design') 

Attention <- Design %>%    
    layer_dense(units=qqq[2], activation='tanh', name='layer1') %>%
    layer_dense(units=qqq[3], activation='tanh', name='layer2') %>%
    layer_dense(units=qqq[4], activation='tanh', name='layer3') %>%
    layer_dense(units=qqq[1], activation='linear', name='attention')

Output <- list(Design, Attention) %>% layer_dot(name='LocalGLM', axes=1) %>% 
    layer_dense(
      units=1, activation='exponential', name='output',
      weights=list(array(0, dim=c(1,1)), array(log_size_hom, dim=c(1)))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training with MLflow
# MAGIC Using MLflow, we can track and log the parameters and metrics associated with this experiment. First we point to our mlflow experiment via the experiment id, then use the `mlflow_start_run()` method to instantiate our logging. After executing this cell, navigate to your mlflow experiments to view the tracked runs. 
# MAGIC 
# MAGIC Before executing the next cell, please create an MLflow experiment and copy the experiment id in the first line of the cell.

# COMMAND ----------

mlflow_set_experiment(experiment_id = "83447046007322") #Update with your experiment id


with(mlflow_start_run(), {
  ## -----------------------------------------------------------------------------
  model_lgn_p2 <- keras_model(inputs = list(Design), outputs = c(Output))

  
  ## -----------------------------------------------------------------------------
  model_lgn_p2 %>% compile(
      loss = 'mse',
      optimizer = 'nadam'
  )
  summary(model_lgn_p2)
  ## -----------------------------------------------------------------------------
  # set hyperparameters
  epochs <- 100
  batch_size <- 5000
  validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
  verbose <- 1
  mlflow_log_param("epochs", epochs)
  mlflow_log_param("batch_size", batch_size)
  mlflow_log_param("validation_split", validation_split)
  ## -----------------------------------------------------------------------------
  # store and use only the best model
  cp_path <- paste("/dbfs/tmp/Networks/model_lgn_p2")

  cp_callback <- callback_model_checkpoint(
      filepath = cp_path,
      monitor = "val_loss",
      save_weights_only = TRUE,
      save_best_only = TRUE,
      verbose = 0
  )


  ## -----------------------------------------------------------------------------
  fit_lgn_p2 <- model_lgn_p2 %>% fit(
      list(XX), list(YY),
      validation_split = validation_split,
      epochs = epochs,
      batch_size = batch_size,
      callbacks = list(cp_callback),
      verbose = verbose
  )

  mlflow_log_model(model_lgn_p2, "model")
  
  ## -----------------------------------------------------------------------------
  plot(fit_lgn_p2) 
  ## -----------------------------------------------------------------------------
  
  keras_plot_loss_min(fit_lgn_p2, seed)
  mlflow_log_artifact("/dbfs/tmp/keras_plot_loss.png") 
  

  ## -----------------------------------------------------------------------------
  load_model_weights_hdf5(model_lgn_p2, cp_path)
  ## -----------------------------------------------------------------------------
  # calculating the predictions
  learn$fitlgnp2 <- as.vector(model_lgn_p2 %>% predict(list(XX)))
  test$fitlgnp2 <- as.vector(model_lgn_p2 %>% predict(list(TT)))

  # average in-sample and out-of-sample losses (in 10^(0))
  sprintf("Gamma deviance shallow network (train): %s", round(gamma_loss(learn$Claim, learn$fitlgnp2), 4))
  sprintf("Gamma deviance shallow network (test): %s", round(gamma_loss(test$Claim, test$fitlgnp2), 4))

  # average claims size
  sprintf("Average size (test): %s", round(mean(test$fitlgnp2), 1))

  mlflow_log_metric("Gamma deviance shallow network (train):", round(gamma_loss(learn$Claim, learn$fitlgnp2), 4))
  mlflow_log_metric("Gamma deviance shallow network (test)", round(gamma_loss(test$Claim, test$fitlgnp2), 4))
  mlflow_log_metric("Average size (test)", round(mean(test$fitlgnp2), 1))
  
  ## ----------------------------------------------------------------------------
  df_cmp %<>% bind_rows(
    data.frame(model = "LocalGLMnet p2 (gamma)",
               learn_p2 = round(gamma_loss(learn$Claim, learn$fitlgnp2), 4),
               learn_pp = round(p_loss(learn$Claim, learn$fitlgnp2, p) * 10, 4),
               learn_p3 = round(ig_loss(learn$Claim, learn$fitlgnp2) * 1000, 4),
               test_p2 = round(gamma_loss(test$Claim, test$fitlgnp2), 4),
               test_pp = round(p_loss(test$Claim, test$fitlgnp2, p) * 10, 4),
               test_p3 = round(ig_loss(test$Claim, test$fitlgnp2) * 1000, 4),
               avg_size = round(mean(test$fitlgnp2), 0)
    ))
  df_cmp
})

# COMMAND ----------


