suppressMessages(library(glmnet))
args = commandArgs(trailingOnly=TRUE)
bp <- args[1]


traind <- as.matrix(read.csv(file=paste(bp, "TRAIN.SF", sep=""), header=FALSE,sep=","))
y <- traind[,1]
x <- traind[,-1]

if (length(unique(y))>2){
  fam <- "multinomial"
}else{
  fam <- "binomial"
}

#cvfit = cv.glmnet(x, y, family=fam, type.measure= "class")

testd <- as.matrix(read.csv(file=paste(bp, "TEST.SF", sep=""), header=FALSE,sep=","))
ty <- testd[,1]
tx <- testd[,-1]

p <- predict(cvfit, newx =tx, type="class", s='lambda.min')

pm <- as.numeric(p)
cm = as.matrix(table(Actual = ty, Predicted = pm))
print(cm)
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted cl
accuracy <- sum(diag) / n
print(accuracy)
error <- 1- sum(diag)/n
print(error)
