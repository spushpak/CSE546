# library(tidyquant)
# library(XLConnect)
# 
# tq_index("SP500") %>% tq_get(get = "stock.prices")

rm(list = ls())

library(xts)

dir <- "C:\\GoogleDrivePushpakUW\\UW\\6thYear\\CSE546\\Project"
setwd(dir)

folder <- "C:\\local\\sandp500\\individual_stocks_5yr\\"    
file_list <- list.files(path=folder, pattern="*.csv") 

file_list

comb_dat <- NULL

# read in each .csv file in file_list and create a data frame with the same name as the .csv file
for (i in 1:length(file_list)){
#for (i in 1:100){
  #assign(file_list[i], read.csv(paste(folder, file_list[i], sep='')))
  temp <- read.csv(paste(folder, file_list[i], sep=''), stringsAsFactors = F)
  ticker <- temp[1, "Name"]
  temp <- xts(temp[, "close"], order.by = as.Date(temp[, "date"]))
  colnames(temp) <- ticker
  comb_dat <- cbind(comb_dat, temp)
  }

head(comb_dat)
tail(comb_dat)
ncol(comb_dat)
nrow(comb_dat)
length(index(comb_dat))
comb_dat[1:100, ]

# Find which firms have missing data
na_count <- colSums(is.na(comb_dat))
str(na_count)

length(na_count[na_count != 0])   # these firms have missing data
miss_val_firms <- names(na_count[na_count != 0])
miss_val_firms

ncol(comb_dat)
str(comb_dat)

retain_firms <- setdiff(colnames(comb_dat), miss_val_firms)
length(retain_firms)

comb_dat <- comb_dat[, retain_firms]

dates <- as.character(as.Date(index(comb_dat)))
sp_dat <- data.frame(cbind(dates, coredata(comb_dat)))
colnames(sp_dat) <- c("date", colnames(comb_dat))


class(comb_dat)
ncol(comb_dat)
nrow(comb_dat)
head(comb_dat)
tail(sp_dat)

save(sp_dat, file = "C:/local/sandp500/sp500.RData")
write.csv(sp_dat, file = "C:/local/sandp500/sp470.csv", row.names = F)



############################################################################
rm(list = ls())

dir <- "C:\\GoogleDrivePushpakUW\\UW\\6thYear\\CSE546\\Project"
setwd(dir)

load("C:/local/sandp500/sp500.RData")

head(sp_dat)

# Tesing glasso code
library(glasso)

set.seed(100)
x<-matrix(rnorm(50*5),ncol=5)
s<- var(x)
a<-glasso(s, rho=.01)
aa<-glasso(s,rho= 0.1, w.init=a$w, wi.init=a$wi)

aa


rho_list <- seq(0.01, 5, 0.1)
glassopath(s, rholist = rho_list, w.init=a$w, wi.init=a$wi)




# example with structural zeros and no regularization,
# from Whittaker's Graphical models book page xxx.
s=c(10,1,5,4,10,2,6,10,3,10)
S=matrix(0,nrow=4,ncol=4)
S[row(S)>=col(S)]=s
S=(S+t(S))
diag(S)<-10
zero<-matrix(c(1,3,2,4),ncol=2,byrow=TRUE)
a<-glasso(S,0,zero=zero)






