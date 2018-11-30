#!/usr/bin/env Rscript
require(argparse)
psr <- ArgumentParser(description="coauthor glue baseline")
psr$add_argument("ipt", help="input", nargs="+")
psr$add_argument("-o", dest="opt", help="output")

args <- psr$parse_args()

library(igraph)
library(dplyr)
library(stringr)

#path <- '/Users/ZLab/Documents/Dropbox/test'
#setwd(paste0(path,'/author0'))
#pv <- list.files()

author<-read.csv(args$ipt) %>% select(-X)
names(author) <- c('name','org','id','seq')

auname <- author %>% group_by(name) %>% dplyr::summarise(count = n()) %>% arrange(desc(count))
auname <- auname$name[1]        
# 生成一个邻接矩阵
node <- unique(author$id)
adjacency <- merge(node,node,ALL=T)
names(adjacency) <- c('node1','node2')
adjacency <- adjacency %>% mutate(node1 = as.character(node1),
                                  node2 = as.character(node2)) %>% 
        filter(node1<node2) %>% mutate(merge=0)

N <- length(unique(adjacency$node1))

for(j in 1:N){
        id_1 <- unique(adjacency$node1)[j]
        name_1 <- author$name[author$id == id_1]
        name_1 <- name_1[name_1!=auname]
        adjacency$merge[adjacency$node1 == id_1&adjacency$node2 %in%
                                author$id[author$name %in% name_1]] <-1
        #cat(j,'\n')
}
adjacency_1 <- adjacency[adjacency$merge==1,]
e <- data.frame(from = adjacency_1$node1,
                to = adjacency_1$node2)

net <- graph_from_data_frame(e, directed=F, vertices=node)

net_c <- components(net)$membership
rst <- data.frame(id=names(net_c), group_result=net_c)
rownames(rst) <- NULL
write.csv(rst,file=args$opt)
