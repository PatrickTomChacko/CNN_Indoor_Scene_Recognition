#Plotting the dataset available
set.seed(1234)
library('keras')
library("jpeg")
library('tfruns')
library("reticulate")
path <- getwd()
path <- paste0(path,"/data_indoor")

path_train = paste0(path,"/train")
fold_names <- list.files(path = path_train)

class_labels <- list()

for(name in fold_names){
  class_labels <- c(class_labels,name)
}

k = 1
par(mfrow = c(2,length(fold_names)/2), mar=rep(1, 4))
for (scene in fold_names) {
     path_scene =  paste0(path_train,"/",scene)
     set = sample(list.files(path=path_scene),1)

     img = readJPEG(paste0(path_scene,"/",set))
     plot(0:1, 0:1, type = "n", ann = FALSE, axes = FALSE)
     rasterImage(img, 0, 0, 1, 1)
     mtext(class_labels[k], line = 0)
  
     k=k+1
    }

set.seed(1234)
val_datagen <- train_datagen <- image_data_generator(rescale = 1/255)
train_generator <- flow_images_from_directory(
path_train,
train_datagen,
target_size = c(64, 64),
batch_size = 20
)

path_val = paste0(path,"/validation") 

val_generator <- flow_images_from_directory(
path_val,
val_datagen,
target_size = c(64, 64),
batch_size = 20
)



#Since CNN's work better with image classifications, I am going to use 4 CNNs of varying hyperparameters, keeping the same number of units and layers.


set.seed(1234)
#CNN model 1 - basic
model1 <- keras_model_sequential() %>%
#
# convolutional layers
layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
input_shape = c(64, 64, 3)) %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#
# fully connected layers
layer_flatten() %>%
layer_dense(units = 256, activation = "relu") %>%
layer_dense(units = 10, activation = "softmax") %>%
#
# compile
compile(
  loss = "categorical_crossentropy",
metrics = "accuracy",
optimizer = optimizer_rmsprop(learning_rate = 0.0001)
)


### Model1
# Our model 1 is a CNN of 4 convolutional layers and 2 fully connected layers, we are going to train it for 100 epochs.

fit1 <- model1 %>% fit(
train_generator,
steps_per_epoch = 20,
epochs = 100,
validation_data = val_generator,
validation_steps = 25
)


### Model 2 
#Model 2 uses the same CNN framework of Model 1, it just differs in the number of units in fully connected layer, changing 1 parameter at a time helps us to understand the impact of that parameter. Model 2 has increased the units by 2 times, higher learning time would help model understand better at the cost of computational time.


#CNN model 2 - 
set.seed(1234)
model2 <- keras_model_sequential() %>%
#
# convolutional layers
layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
input_shape = c(64, 64, 3)) %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#
# fully connected layers
layer_flatten() %>%
layer_dense(units = 512, activation = "relu") %>%          #change
layer_dense(units = 10, activation = "softmax") %>%
#
# compile
compile(
  loss = "categorical_crossentropy",
metrics = "accuracy",
optimizer = optimizer_rmsprop(learning_rate = 0.0001)
)

#
fit2 <- model2 %>% fit(
train_generator,
steps_per_epoch = 20,
epochs = 100,
validation_data = val_generator,
validation_steps = 25
)


# We can see during the plot, the training has reached a maximum accuracy over 0.7 but the corresponding validation accuracy is 0.38, which simply means our model is learning more but it overfitting, hence we need to penalise the overfitting for better general predictive performance.

### Model 3
# We are going to futher develop on Model2, we are going to add L2 regularization and dropout both of arguements 0.1, we expect the Model3 to be better as it is going to penalize overfitting.


set.seed(1234)
#CNN model 3 (Model 2 + Regularization + drop out)
model3 <- keras_model_sequential() %>%
#
# convolutional layers
layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
input_shape = c(64, 64, 3)) %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#
# fully connected layers
layer_flatten() %>%
layer_dense(units = 256, activation = "relu", kernel_regularizer = regularizer_l2(0.1)) %>%
layer_dropout(0.1) %>%
layer_dense(units = 10, activation = "softmax") %>%
#
# compile
compile(
  loss = "categorical_crossentropy",
metrics = "accuracy",
optimizer = optimizer_rmsprop(learning_rate = 0.0001)        #lower learning rate, more general
)


fit3 <- model3 %>% fit(
train_generator,
steps_per_epoch = 20,
epochs = 100,
validation_data = val_generator,
validation_steps = 25
)

We can see from the plot that the gap between training and validation accuracy has decreased and both curves are close together, so our model is no longer overfitting.


### Model 4
# We are going to add Batch Normalization such that our parameter estimates are more stable and convergence is reached faster. We keep the epochs fixed to 100 as we are giving more time for our model to learn.

set.seed(1234)
#CNN model 4 (Model 3 + Batch normalization)
model4 <- keras_model_sequential() %>%

# convolutional layers
layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
input_shape = c(64, 64, 3)) %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_batch_normalization() %>%
layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_batch_normalization() %>%
layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#
# fully connected layers
layer_flatten() %>%
layer_dense(units = 256, activation = "relu", kernel_regularizer = regularizer_l2(1)) %>%
layer_dropout(0.1) %>%
layer_dense(units = 10, activation = "softmax") %>%
#
# compile
compile(
  loss = "categorical_crossentropy",
metrics = "accuracy",
optimizer = optimizer_rmsprop(learning_rate = 0.0001)        #lower learning rate, more general
)

fit4 <- model4 %>% fit(
train_generator,
steps_per_epoch = 20,
epochs = 100,
validation_data = val_generator,
validation_steps = 25
)


# to add a smooth line to points
smooth_line <- function(y) {
x <- 1:length(y)
out <- predict( loess(y ~ x) )
return(out)
}


### Model 1
```{r}
# check learning curves
cols <- c("black", "dodgerblue3")
out1 <- cbind(fit1$metrics$loss,
fit1$metrics$val_loss,
fit1$metrics$accuracy,
fit1$metrics$val_accuracy)
```


```{r}
i =0
sum =0
while(i!=20){
       val_acc =  evaluate(model1, val_generator, steps = length(val_generator)/20)
       sum = sum+val_acc
       i = i+1
}
sum/20



### Model 2
```{r}
cols <- c("black", "dodgerblue3")
out2 <- cbind(fit2$metrics$loss,
fit2$metrics$val_loss,
fit2$metrics$accuracy,
fit2$metrics$val_accuracy)


i =0
sum =0
while(i!=20){
       val_acc =  evaluate(model2, val_generator, steps = length(val_generator)/20)
       sum = sum+val_acc
       i = i+1
}
sum/20


### Model 3

cols <- c("black", "dodgerblue3")
out3 <- cbind(fit3$metrics$loss,
fit3$metrics$val_loss,
fit3$metrics$accuracy,
fit3$metrics$val_accuracy)


i =0
sum =0
while(i!=20){
       val_acc =  evaluate(model3, val_generator, steps = length(val_generator)/20)
       sum = sum+val_acc
       i = i+1
}
sum/20

### Model 4
cols <- c("black", "dodgerblue3")
out4 <- cbind(fit4$metrics$loss,
fit4$metrics$val_loss,
fit4$metrics$accuracy,
fit4$metrics$val_accuracy)

i =0
sum =0
while(i!=20){
       val_acc =  evaluate(model4, val_generator, steps = length(val_generator)/20)
       sum = sum+val_acc
       i = i+1
}
sum/20

#Here we used batch normalization and a huge penalty of 1 in L2 regularization to train our model, the results are better, we got a predictive performance of 0.41 in the same dataset used for all the above models. Thus we go forward with this model and discard the other models, Due to less number of dataset in some scenes the model must have overfitted, for which huge penalty made it better, Data augmentation can be used to generate more samples in such case or we can club together scenes that look similiar.

#### Plot Inference
par(mfrow = c(1,4))
# accuracy
matplot(out1[,3:4], pch = 19, ylab = "Loss", xlab = "Epochs",
col = adjustcolor(cols, 0.3), ylim = c(0, 1), main = "Model1")
matlines(apply(out1[,3:4], 2, smooth_line), lty = 1, col = cols, lwd = 2)
legend("topright", legend = c("Training", "Validation"),
fill = cols, bty = "n")
# accuracy 2
matplot(out2[,3:4], pch = 19, ylab = "Loss", xlab = "Epochs",
col = adjustcolor(cols, 0.3), ylim = c(0, 1),  main = "Model2")
matlines(apply(out2[,3:4], 2, smooth_line), lty = 1, col = cols, lwd = 2)
legend("topright", legend = c("Training", "Validation"),
fill = cols, bty = "n")

# accuracy Model 3
matplot(out3[,3:4], pch = 19, ylab = "Loss", xlab = "Epochs",
col = adjustcolor(cols, 0.3), ylim = c(0, 1),  main = "Model3")
matlines(apply(out3[,3:4], 2, smooth_line), lty = 1, col = cols, lwd = 2)
legend("topright", legend = c("Training", "Validation"),
fill = cols, bty = "n")

# accuracy Model 4
matplot(out4[,3:4], pch = 19, ylab = "Loss", xlab = "Epochs",
col = adjustcolor(cols, 0.3), ylim = c(0, 1),  main = "Model4")

matlines(apply(out4[,3:4], 2, smooth_line), lty = 1, col = cols, lwd = 2)
legend("topright", legend = c("Training", "Validation"),
fill = cols, bty = "n")
```

# We can see here that Model 4 shows huge signs of overfitting of training data, but considering the validation accuracy its still slightly over the other models, so in this case I neglect the signs of overfitting as it is still able to yield high general predictive performance as considered to others.

### Testing
evaluate(model1, test_generator)
set.seed(1234)
pred <- predict(model4,test_generator)
pred_class <- max.col(pred)-1
y_act <-test_generator$classes
y_act
table(pred_class,y_act)