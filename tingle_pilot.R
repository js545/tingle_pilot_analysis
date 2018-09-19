# Jake Son
# Child Mind Institute

library(mgc)
library(reshape2)
library(ggplot2)

plot_mtx <- function(Dx, main.title="Distance Matrix",
                     xlab.title="Sample Sorted by Source",
                     ylab.title="Sample Sorted by Source") {
  data <- melt(Dx)
  ggplot(data, aes(x=Var1, y=Var2, fill=value)) +
    geom_tile() +
    scale_fill_gradientn(name="dist(x, y)",
                         colours=c("#f2f0f7", "#cbc9e2", "#9e9ac8", "#6a51a3"),
                         limits=c(min(Dx), max(Dx))) +
    xlab(xlab.title) +
    ylab(ylab.title) +
    theme_bw() +
    ggtitle(main.title)
}

# DISCRIMINABILITY ANALYSIS

setwd('/Users/jakeson/Documents/CMI/Tingle Device/')
data = read.csv('tingle_pilot_data_shareable.csv')

vars = c('distance', 'pitch', 'roll',
         'thermopile1', 'thermopile2', 'thermopile3', 'thermopile4',
         'target', 'ontarget',
         'participant')

mydata <- data[vars]

targets = c('rotate-mouth',
            'rotate-nose',
            'rotate-cheek',
            'rotate-eyebrow',
            'rotate-top-head',
            'rotate-back-head')

mydata <- subset(mydata, target %in% targets)

mydata <- subset(mydata, ontarget == 'True')

# # Participant 1
# p1data <- subset(mydata, participant == 1)
# p1data <- droplevels(p1data)
# 
# # Check number of samples per condition
# table(unlist(p1data$target))
# 
# # Test discriminability score
# 
# rownames(p1data) <- 1:nrow(p1data)
# Dx <- as.matrix(dist(p1data[sort(as.vector(p1data$target),
#                                  index=TRUE)$ix,c(1,2,3,4,5,6,7)]))
# plot_mtx(Dx)
# 
# # Calculate discriminability score
# discr.stat(p1data[,c(1,2,3,4,5,6,7)], as.vector(p1data$target))

# All participants, individual

iter <- 50
results <- matrix(NA, nrow=iter, ncol=8)
colnames(results) = c('Participant', 'Discrim_Score', 'N_back_head', 'N_cheek', 'N_eyebrow', 'N_mouth',
                      'N_nose', 'N_top_head')

for (participant_num in unique(mydata$participant)){
  
  pdata <- subset(mydata, participant == participant_num)
  pdata <- droplevels(pdata)
  
  out <- table(unlist(pdata$target))
  back_head = out[1]
  cheek = out[2]
  eyebrow = out[3]
  mouth = out[4]
  nose = out[5]
  top_head = out[6]
  
  discr_score = discr.stat(pdata[,c(1,2,3,4,5,6,7)], as.vector(pdata$target))
  print(discr_score)
  
  results[participant_num,] = c(participant_num, discr_score, out[1],out[2],out[3],out[4],out[5],out[6])
  
}

# All participants, group

groupdata = droplevels(mydata)
out <- table(unlist(groupdata$target))
group_discr_score = discr.stat(groupdata[,c(1,2,3,4,5,6,7)], as.vector(pdata$target))

results[participant_num+1,] = c(100, group_discr_score, out[1],out[2],out[3],out[4],out[5],out[6])

results = na.omit(results)

write.csv(results, file='discr_results.csv')

# MULTISCALE GRAPH CORRELATION ANALYSIS

dyads <- combn(targets, m=2, simplify=TRUE)

cnames = list()

for (num in 1:dim(dyads)[2]){
  
  dyad = paste(dyads[,num], collapse = '_')
  cnames <- append(cnames, dyad)
  
}

iter <- 50
results <- matrix(NA, nrow=iter, ncol=dim(dyads)[2])
colnames(results) = cnames

# Participant 1

p1data = subset(mydata, participant == 1)

for (participant_num in unique(mydata$participant)){
# for (participant_num in 1:2){
  print(paste(c('Analyzing participant', participant_num), collapse = ' '))

  for (num in 1:dim(dyads)[2]){
    
    prop1 <- subset(p1data, target == dyads[,num][1])
    prop2 <- subset(p1data, target == dyads[,num][2])
    
    prop1 <- data.matrix(prop1[1:7], rownames.force=NA)
    prop2 <- data.matrix(prop2[1:7], rownames.force=NA)
    
    output <- mgc.test(nose, cheek, rep=1000)
    
    pMGC = output$pMGC
    
    results[participant_num,num] <- pMGC
    
  }
  
}

results = na.omit(results)

head(results, n=3)

write.csv(results, file='mgc_results.csv')

# NEURAL NETWORK ANALYSIS

library(kerasR)
library(reticulate)

use_python("/Users/jakeson/anaconda3/bin/python3.6")
keras_init()
keras_available()







