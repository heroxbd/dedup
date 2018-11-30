#coauthor glue 

#!/usr/bin/env Rscript
require(argparse)
psr <- ArgumentParser(description="coauthor glue baseline")
psr$add_argument("ipt", help="input", nargs="+")
psr$add_argument("-o", dest="opt", help="output")

args <- psr$parse_args()


library(igraph)
library(dplyr)
library(rjson)
library(stringr)
require(plyr)
library(rhdf5)
library(tidyr)

author<-read.csv(pv[i]) %>% select(-X)
names(author) <- c('name','org','id','seq')
fname <- str_replace_all(pv[i],pattern='.csv',replacement = '')
auname <- author %>% group_by(name) %>% dplyr::summarise(count = n()) %>% arrange(desc(count))
auname <- auname$name[1]


# 将所有的article形成一个矩阵
node_au <- as.character(unique(author$id))
node_coau <- unique(author$name)[unique(author$name)!=auname]
node_all <- c(node_au,node_coau)

adjacency <- merge(node_all,node_coau,ALL=T)
names(adjacency) <- c('node1','node2')
adjacency <- adjacency %>% mutate(node1 = as.character(node1),
                                  node2 = as.character(node2)) %>% 
        filter(node1<node2) %>% mutate(merge=0)

N <- length(unique(node_au))

start <- Sys.time()
for(j in 1:N){
        name_1 <- author$name[author$id==node_au[j]]
        name_2 <- name_1[name_1!=auname]
        adjacency$merge[adjacency$node1 %in% c(node_au[j],name_2) & adjacency$node2 %in% name_2] <- 1
        adjacency[adjacency$node1 %in% c(node_au[j],name_2) & adjacency$node2 %in% name_2,]
        cat(j,'\n')
}
end <- Sys.time()
end-start 

#提取出有效连边
adjacency_1 <- adjacency[adjacency$merge==1,]
e <- data.frame(from = adjacency_1$node1,
                to = adjacency_1$node2)
node <- unique(c(adjacency_1$node1,adjacency_1$node2))
# 生成图
net <- graph_from_data_frame(e, directed=F, vertices=node)

# 计算节点间距离
start <- Sys.time()
dist <- distances(net)

# 变形成为similarity 


dist_final <- data.frame(dist)
dist_final <- dist_final[rownames(dist_final)%in% node_au,]
dist_final$node1 <- rownames(dist_final)
dist_final <- dist_final %>% gather(node2, value, -node1) %>% mutate(node2 = str_remove(node2,pattern = 'X')) %>%
        filter(node2 %in% node_au)
# 输出数据
write_json(dist_final,path = paste0(args$opt,'/result/',fname,'.json'))








