#!/usr/bin/env Rscript
require(argparse)
require(plyr)

psr <- ArgumentParser(description="combine id title abstract")
psr$add_argument("ipt", nargs="+", help="item input")
psr$add_argument("-o", dest="opt", help="output")
psr$add_argument('--abstract', nargs="+", help="abstract input")
args <- psr$parse_args()

item <- ldply(args$ipt, function(f) read.csv(f, row.names=1))

abstract <- NULL
try(abstract <- ldply(args$abstract, function(f) read.csv(f, row.names=1)))
if(is.null(abstract)){
    ia <- item
    ia$abstract <- NA
} else {
    ia <- merge(item, abstract, all.x=TRUE)
}
write.csv(ia[c('id', 'auid', 'title', 'abstract')], file=args$opt)
