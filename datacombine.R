rm(list=ls())
#install.packages('rjson')
library(rjson)
library(stringr)
setwd('/Users/ZLab/Downloads')

# 合并文件
item_final <- data.frame()
author_final <- data.frame()
abstract_final <- data.frame()
keywords_final <- data.frame()
files <- list.files()

# item
itemcsvfile <- files[str_detect(files,pattern='^item[0-9]+[//.]csv')]
for(i in itemcsvfile){
        tmp <- read.csv(file = i)
        tmp$X <- NULL
        item_final <- rbind(item_final,tmp)
}

#abstract
abcsvfile <- files[str_detect(files,pattern='^abstract[0-9]+[//.]csv')]
for(i in abcsvfile){
        tmp <- read.csv(file = i)
        tmp$X <- NULL
        tmp[,1] <- as.character(tmp[,1])
        tmp[,2] <- as.character(tmp[,2])
        abstract_final <- rbind(abstract_final,tmp)
}
#authors
aucsvfile <- files[str_detect(files,pattern='^author[0-9]+[//.]csv')]
for(i in aucsvfile){
        tmp <- read.csv(file = i)
        tmp$X <- NULL
        tmp[,1] <- as.character(tmp[,1])
        tmp[,2] <- as.character(tmp[,2])
        author_final <- rbind(author_final,tmp)
}

#keywords
keycsvfile <- files[str_detect(files,pattern='^keywords[0-9]+[//.]csv')]
for(i in keycsvfile){
        tmp <- read.csv(file = i)
        tmp$X <- NULL
        tmp[,1] <- as.character(tmp[,1])
        tmp[,2] <- as.character(tmp[,2])
        keywords_final <- rbind(keywords_final,tmp)
}

write.csv(item_final,file='item_final.csv',row.names = FALSE)
write.csv(author_final,file='author_final.csv',row.names = FALSE)
write.csv(abstract_final,file='abstract_final.csv',row.names = FALSE)
write.csv(keywords_final,file='keywords_final.csv',row.names = FALSE)