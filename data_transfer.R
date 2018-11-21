#!/usr/bin/env Rscript
require(argparse)

psr <- ArgumentParser(description="convert variable-lengthed json into long form")
psr$add_argument("ipt", help="input", nargs="+")
psr$add_argument("-o", dest="opt", help="output")

args <- psr$parse_args()

library(rjson)
require(plyr)
require(doMC)
registerDoMC()

pv <- fromJSON(file = args$ipt)

datatrans <-function(i){
    tmp <- pv[i]
    disamau <- names(tmp)
    tmp1 <- tmp[[1]]
    item <- data.frame()
    author <- data.frame()
    abstract <- data.frame()
    keywords <- data.frame()

    item <- ldply(tmp1, function(x) as.data.frame(x[c('id','title','venue','year')]))
    item$auid <- disamau

    abstract <- ldply(tmp1, function(x) 
        if ("abstract" %in% names(x)) {
            as.data.frame(x[c('id','abstract')])
        })

    keywords <- ldply(tmp1, function(x) {
        d <- data.frame(keywords=x$keywords)
        d$id <- x$id
        d
    })

    author <- ldply(tmp1, function(x) {
        d <- ldply(x$authors, data.frame)
        d$id <- x$id
        d$auseq <- rownames(d)
        d
    })

    write.csv(item,file=paste0(args$opt,'/item/',disamau,'.csv'))
    write.csv(abstract,file=paste0(args$opt,'/abstract/',disamau,'.csv'))
    write.csv(author,file=paste0(args$opt,'/author/',disamau,'.csv'))
    write.csv(keywords,file=paste0(args$opt,'/keywords/',disamau,'.csv'))
}

l_ply(1:length(pv), datatrans, .parallel=TRUE)
