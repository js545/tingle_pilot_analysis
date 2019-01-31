# Jake Son
# Child Mind Institute

library(mgc)
library(reshape2)
library(ggplot2)
library(ggridges)

# Ridgeline plots

setwd('/Users/jakeson/Documents/CMI/tingle_pilot_analysis/results')
n_data = read.csv('auroc_n_thermal.csv')
y_data = read.csv('auroc_y_thermal.csv')
n_data$Thermal = c(rep('No thermal', dim(n_data)[1]))
y_data$Thermal = c(rep('Yes thermal', dim(y_data)[1]))

data = rbind(n_data, y_data)
data$thermal = factor(data$Thermal)

data = subset(data, Target == 'mouth')
data = subset(data, Target == 'nose')
data = subset(data, Target == 'cheek')
data = subset(data, Target == 'eyebrow')
data = subset(data, Target == 'top-head')
data = subset(data, Target == 'back-head')

for (tar in unique(data$Target)) {
  
  png_data = subset(data, Target == tar)
  
  savename = paste('~/Documents/CMI/tingle_pilot_analysis/ridgeline_individual_plots/', tar, '.png', sep="")
  
  savename
  
  png(savename, width=1200, height=1200, res=600)
  print(ggplot(png_data, aes(x=AUROC, y=Target, fill=Thermal, point_color=Thermal, height=..density../39)) + 
    scale_fill_cyclical(values = c("blue", "red")) + 
    # geom_density_ridges(jittered_points=TRUE, point_size=3, size=.25, point_shape="|",
    # position=position_points_jitter(height=0), alpha=.5, scale=.95) +
    geom_density(aes(y= ..density../39), stat='density', alpha=.5) +
    scale_discrete_manual("point_color", values= c('blue', 'red')) + 
    theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + 
    ylim(0, .7))
  dev.off()
  
}


# Analysis

setwd('/Users/jakeson/Documents/CMI/tingle_pilot_analysis/')
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

# # DISCRIMINABILITY ANALYSIS
# 
# # All participants, individual
# 
# iter <- 50
# results <- matrix(NA, nrow=iter, ncol=8)
# colnames(results) = c('Participant', 'Discrim_Score', 'N_back_head', 'N_cheek', 'N_eyebrow', 'N_mouth',
#                       'N_nose', 'N_top_head')
# 
# for (participant_num in unique(mydata$participant)){
#   
#   pdata <- subset(mydata, participant == participant_num)
#   pdata <- droplevels(pdata)
#   
#   out <- table(unlist(pdata$target))
#   back_head = out[1]
#   cheek = out[2]
#   eyebrow = out[3]
#   mouth = out[4]
#   nose = out[5]
#   top_head = out[6]
#   
#   # discr_score = discr.stat(pdata[,c(1,2,3,4,5,6,7)], as.vector(pdata$target))
#   discr_score = discr.stat(pdata[,c(1,2,3)], as.vector(pdata$target))
#   print(discr_score)
#   
#   results[participant_num,] = c(participant_num, discr_score, out[1],out[2],out[3],out[4],out[5],out[6])
#   
# }
# 
# # All participants, group
# 
# groupdata = droplevels(mydata)
# out <- table(unlist(groupdata$target))
# # group_discr_score = discr.stat(groupdata[,c(1,2,3,4,5,6,7)], as.vector(pdata$target))
# group_discr_score = discr.stat(groupdata[,c(1,2,3)], as.vector(pdata$target))
# 
# results[participant_num+1,] = c(100, group_discr_score, out[1],out[2],out[3],out[4],out[5],out[6])
# 
# results = na.omit(results)
# 
# write.csv(results, file='discr_results.csv')

# MULTISCALE GRAPH CORRELATION ANALYSIS

iter <- 50
results <- matrix(NA, nrow=iter, ncol=length(targets))
colnames(results) = targets

# for (participant_num in unique(mydata$participant)){
for (participant_num in 23:41) {
  
  if (participant_num==22) next
  
  print(paste(c('Analyzing participant', participant_num), collapse = ' '))
  
  p1data = subset(mydata, participant == participant_num)
  tar_int = 1
  
  # for (num in 1:dim(dyads)[2]){
  for (tar in targets){
    
    print(tar)
    
    prop1 <- subset(p1data, target == tar)
    prop2 <- subset(p1data, target != tar)
    
    # prop1 <- data.matrix(prop1[1:7], rownames.force=NA)
    # prop2 <- data.matrix(prop2[1:7], rownames.force=NA)
    
    prop1 <- data.matrix(prop1[1:3], rownames.force=NA)
    prop2 <- data.matrix(prop2[1:3], rownames.force=NA)
    
    rownames(prop1) <- NULL
    rownames(prop2) <- NULL
    colnames(prop1) <- NULL
    colnames(prop2) <- NULL
    
    x_data = rbind(prop1, prop2)
    y_data = rbind(matrix(1, dim(prop1)[1], 1), matrix(0, dim(prop2)[1], 1))
    
    # y_data <- y_data[sample(nrow(y_data), nrow(y_data)), ]
    
    output <- mgc.test(x_data, y_data, rep=500)
    
    pMGC = output$pMGC
    
    print(pMGC)
    
    results[participant_num,tar_int] <- pMGC
    
    tar_int <- tar_int +1
    
  }
  
}

results = na.omit(results)

# head(results, n=3)

write.csv(results, file='binarized_mgc_wo_temp_41.csv')

# # MULTISCALE GRAPH CORRELATION ANALYSIS

dyads <- combn(targets, m=2, simplify=TRUE)

cnames = list()

for (num in 1:dim(dyads)[2]){

  dyad = paste(dyads[,num], collapse = '_')
  cnames <- append(cnames, dyad)

}

iter <- 50
results <- matrix(NA, nrow=iter, ncol=dim(dyads)[2])
colnames(results) = cnames

for (participant_num in unique(mydata$participant)){

# for (participant_num in 1:3){
  
  print(paste(c('Analyzing participant', participant_num), collapse = ' '))

  p1data = subset(mydata, participant == participant_num)

  for (num in 1:dim(dyads)[2]){
  # for (num in 1:5){

    prop1 <- subset(p1data, target == dyads[,num][1])
    prop2 <- subset(p1data, target == dyads[,num][2])

    prop1 <- data.matrix(prop1[1:3], rownames.force=NA)
    prop2 <- data.matrix(prop2[1:3], rownames.force=NA)
    
    rownames(prop1) <- NULL
    rownames(prop2) <- NULL
    colnames(prop1) <- NULL
    colnames(prop2) <- NULL
    
    x_data = rbind(prop1, prop2)
    y_data = rbind(matrix(1, dim(prop1)[1], 1), matrix(0, dim(prop2)[1], 1))

    output <- mgc.test(x_data, y_data, rep=1000)

    pMGC = output$pMGC
    
    print(c(dyads[,num], pMGC))

    results[participant_num,num] <- pMGC

  }

}

results = na.omit(results)

head(results, n=3)

write.csv(results, file='1000_mgc_results_n_thermal.csv')









# Ridgeline plots for sampling distribution

library(mgc)
library(reshape2)
library(ggplot2)
library(ggridges)

setwd('/Users/jakeson/Documents/CMI/tingle_pilot_analysis/results')
n_data = read.csv('auroc_n_thermal.csv')
y_data = read.csv('auroc_y_thermal.csv')
n_data$Thermal = c(rep('No thermal', dim(n_data)[1]))
y_data$Thermal = c(rep('Yes thermal', dim(y_data)[1]))

data = rbind(n_data, y_data)
data$thermal = factor(data$Thermal)

ggplot(data, aes(x=AUROC, y=Target, fill=Thermal, point_color=Thermal)) + scale_fill_cyclical(values = c("blue", "red")) + geom_density_ridges(jittered_points=TRUE, point_size=3, size=.25, point_shape="|", position=position_points_jitter(height=0), alpha=.5, scale=.95) + ggtitle('AUROC Comparison') + theme(plot.title = element_text(hjust = 0.5)) + scale_discrete_manual("point_color", values= c('blue', 'red'))


with_path = '/Users/jakeson/Documents/CMI/tingle_pilot_analysis/permutation_results_w/'
without_path = '/Users/jakeson/Documents/CMI/tingle_pilot_analysis/permutation_results_wo/'

w_data_path = paste(with_path, '1_permutations.csv', sep = "")
wo_data_path = paste(without_path, '1_permutations.csv', sep = "")

n_data = read.csv(w_data_path)
y_data = read.csv(wo_data_path)

n_data$Thermal = c(rep('No thermal', dim(n_data)[1]))
y_data$Thermal = c(rep('Yes thermal', dim(y_data)[1]))
n_data$location = c(rep('Mouth-Nose', dim(n_data)[1]))
y_data$location = c(rep('Mouth-Nose', dim(y_data)[1]))

data = rbind(n_data, y_data)
data$Thermal = factor(data$Thermal)

tar1 = 'rotate.mouth_rotate.cheek'

{ggplot(data, aes(x=rotate.mouth_rotate.cheek, y=location, fill=Thermal, point_color=Thermal)) +
  scale_fill_cyclical(values = c("red", " blue")) +
  geom_density_ridges(jittered_points=FALSE, point_size=3, size=.25, point_shape="|", position=position_points_jitter(height=0), alpha=.5, scale=.95) +
  ggtitle('Distance Comparison') +
  theme(plot.title = element_text(hjust = 0.5))}











