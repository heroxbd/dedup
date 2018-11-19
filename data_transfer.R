library(rjson)
setwd('/Users/ZLab/Downloads')
data <- fromJSON(file = 'pubs_train.json')
item <- data.frame()
author <- data.frame()
abstract <- data.frame()
keywords <- data.frame()
for(i in 1:100){
  tmp <- data[i]
  disamau <- names(tmp)
  tmp1 <- tmp[[1]]
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
  cat(disamau,'\n')
}

