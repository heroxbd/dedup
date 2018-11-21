rm(list=ls())
#install.packages('rjson')
library(rjson)
library(stringr)
setwd('/Users/ZLab/Downloads')
data <- fromJSON(file = 'pubs_train.json')
# 拆分数据
item.list <- list()
author.list <- list()
abstract.list <- list()
keywords.list <- list()

datatrans <-function(i){
        item <- data.frame()
        author <- data.frame()
        abstract <- data.frame()
        keywords <- data.frame()
        # 提取数据
        tmp <- data[i]
        disamau <- names(tmp)
        tmp1 <- tmp[[1]]
        #  做item的表
        for(j in 1:length(tmp1)){
                tmp2 <- tmp1[[j]]
                id <- tmp2$id
                itemname <- c('id','title','venue','year')
                for(key in itemname[!itemname%in%names(tmp2)]){
                        tmp2$new <- ''
                        names(tmp2)[names(tmp2)=='new'] <- key
                   }
                itemtmp <- as.data.frame(tmp2[c('id','title','venue','year')],stringsAsFactors = F)
                itemtmp$auid <- disamau
                item <- rbind(item,itemtmp)
                
                if(length(tmp2$abstract)>0){
                  abtmp <-as.data.frame(tmp2$abstract,stringsAsFactors=F)
                  abtmp$id <- id
                  abstract <- rbind(abstract,abtmp)
                }
                if(length(tmp2$keywords)>0){
                        for (l in 1:length(tmp2$keywords)){
                                keytmp <- as.data.frame(tmp2$keywords[l],stringsAsFactors=F)
                                keytmp$id <- id
                                keywords <- rbind(keywords,keytmp)
                        }
                }
                tmp3 <- tmp2['authors'][[1]]
                for(k in 1:length(tmp3)){
                  autmp <- as.data.frame(tmp3[[k]], stringsAsFactors = F)
                  autmp$id <- id
                  autmp$auseq <- k
                  author <- rbind(author,autmp)
                }
                cat(disamau,'\t',j,'\n')
        }
        
        filename <- paste0('item',i,'.csv')
        write.csv(item,file=filename,row.names = FALSE)
        filename <- paste0('author',i,'.csv')
        write.csv(author,file=filename,row.names = FALSE)
        filename <- paste0('abstract',i,'.csv')
        write.csv(abstract,file=filename,row.names = FALSE)
        filename <- paste0('keywords',i,'.csv')
        write.csv(keywords,file=filename,row.names = FALSE)
}
for(i in 1:100){
        datatrans(i)
}






