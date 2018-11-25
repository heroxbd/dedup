#!/usr/bin/env Rscript
library(igraph)
library(dplyr)
library(stringr)
require(argparse)
psr <- ArgumentParser(description="universal glue")
psr$add_argument("ipt", help="item input")
psr$add_argument("-o", dest="opt", help="output")
psr$add_argument("--author", help="author input")
args <- psr$parse_args()

# load data
item_final <- read.csv(args$ipt, row.names=1, stringsAsFactors=FALSE)
author_final <- read.csv(args$author, row.names=1, stringsAsFactors=FALSE)

# using venue, title, and org as naive groups
# get the 
#key <- item_final$auid[i]
#temp <- item_final[item_final$auid==key,]
temp <- item_final %>% mutate(group1 = as.numeric(as.factor(as.character(item_final$venue))),
                           group2 = as.numeric(as.factor(as.character(item_final$title)))) %>%
                select(id,group1,group2) %>%
        distinct_()
temp2 <- author_final[author_final$id %in% temp$id,] %>% select(id,name,org)
# get the focus author name of the object 
au_name <- names(sort(table(temp2$name),decreasing = T)[1])
# extract the org information of the focus author and then group by
temp2 <- temp2 %>%    
        filter(name == names(sort(table(temp2$name),decreasing = T)[1])) %>%
        distinct_()
temp2$org[temp2$org==''] <- as.character(1:sum(temp2$org=='')) 
temp2 <- temp2 %>% 
    mutate(group3 = as.numeric(as.factor(as.character(temp2$org)))) %>%
    select(id,group3)

if(sum(duplicated(temp2$id))>0){
        du_id <- temp2$id[duplicated(temp2$id)]
        for(i in 1:length(du_id)){
                temp2$id[duplicated(temp2$id)&temp2$id==du_id[i]]<- 
                                paste0(temp2$id[duplicated(temp2$id)&temp2$id==du_id[i]],'_',
                                       1:sum(duplicated(temp2$id)&(temp2$id==du_id[i])))
        }
}

temp3 <- inner_join(temp,temp2,by='id')

# make empty group 留着后面优化
# 造图不知道空的图怎么造，先做最傻的办法
index.df <- temp3 %>% group_by(group1) %>% summarise(count=n()) %>% filter(count>1)
index <- index.df$group1
ve <- as.character(temp3$id[temp3$group1==index[1]])
v <- data.frame(name = as.character(temp3$id))
lg <- length(ve)
e <- data.frame(from = ve[-lg],
                to = ve[-1])
net1 <- graph_from_data_frame(e, directed=F, vertices=v)

# make a subgraph for each group and union them togethor
for (i in index[-1]){
        ve <- as.character(temp3$id[temp3$group1==i])
        v <- data.frame(name = ve)
        lg <- length(ve)
        e <- data.frame(from = ve[-lg],
                        to = ve[-1])
        g <- graph_from_data_frame(e, directed=F, vertices=v)
        net1 <- net1 %u% g
}

#net3
index.df <- temp3 %>% group_by(group3) %>% summarise(count=n()) %>% filter(count>1)
index <- index.df$group3
ve <- as.character(temp3$id[temp3$group3==index[1]])
v <- data.frame(name = as.character(temp3$id))
lg <- length(ve)
e <- data.frame(from = ve[-lg],
                to = ve[-1])
net3 <- graph_from_data_frame(e, directed=F, vertices=v)

for (i in index[-1]){
        ve <- as.character(temp3$id[temp3$group3==i])
        v <- data.frame(name = ve)
        lg <- length(ve)
        e <- data.frame(from = ve[-lg],
                        to = ve[-1])
        g <- graph_from_data_frame(e, directed=F, vertices=v)
        net3 <- net3 %u% g
}

# net2  
index.df <- temp3 %>% group_by(group2) %>% summarise(count=n())
if(max(index.df$count)>1) {
    index.df <- index.df %>% filter(count>1)
    index <- index.df$group2
    ve <- as.character(temp3$id[temp3$group2==index[1]])
    v <- data.frame(name = as.character(temp3$id))
    lg <- length(ve)
    e <- data.frame(from = ve[-lg],
                   to = ve[-1])
    net2 <- graph_from_data_frame(e, directed=F, vertices=v)

    for (i in index[-1]){
        ve <- as.character(temp3$id[temp3$group2==i])
        v <- data.frame(name = ve)
        lg <- length(ve)
        e <- data.frame(from = ve[-lg],
                       to = ve[-1])
        g <- graph_from_data_frame(e, directed=F, vertices=v)
        net2 <- net2 %u% g
    }
    net <- net1 %u% net2 %u% net3
} else {
    net <- net1 %u% net3
}

net_c <- components(net)$membership
temp3$group_result <- net_c

temp3 <- temp3 %>% mutate(id = str_replace_all(id, pattern = '_[0-9]+',replacement = ''))
# output
write.csv(temp3[c('id','group_result')],file = args$opt)
