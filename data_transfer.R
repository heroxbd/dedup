#!/usr/bin/env Rscript
require(argparse)

psr <- ArgumentParser(description="convert variable-lengthed json into long form")
psr$add_argument("ipt", help="input", nargs="+")
psr$add_argument("-o", dest="opt", help="output")

args <- psr$parse_args()

library(rjson)
require(plyr)
require(doMC)
registerDoMC(cores=14)

pv <- fromJSON(file = args$ipt)

datatrans <-function(i){
    tmp <- pv[i]
    item <- data.frame()
    author <- data.frame()
    abstract <- data.frame()
    keywords <- data.frame()
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
    }

    write.table(item,file=paste0('data/tabular/item/',disamau,'.csv'),sep=',',col.names=FALSE)
    write.table(author,file=paste0('data/tabular/author/',disamau,'.csv'),sep=',',col.names=FALSE)
    write.table(abstract,file=paste0('data/tabular/abstract/',disamau,'.csv'),sep=',',col.names=FALSE)
    write.table(keywords,file=paste0('data/tabular/keywords/',disamau,'.csv'),sep=',',col.names=FALSE)
}

l_ply(1:length(pv), datatrans, .parallel=TRUE)
