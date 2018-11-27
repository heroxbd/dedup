## define author family name 
library(stringr)
library(dplyr)
setwd("/Users/ZLab/Downloads")
author <- read.csv('./aminer/author_final.csv')
author_list <- str_split(author$name,pattern = '[:blank:]')
last_name <- unlist(lapply(author_list,function(x) tail(x,1)))
last_name <- data.frame(last_name,stringsAsFactors = F)
last_name$id <- author$id

write.csv(last_name, file='last_name.csv')