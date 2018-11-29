#coauthor glue 

library(igraph)
library(dplyr)
library(rjson)
library(stringr)

path <- '/Users/ZLab/Documents/Dropbox/test'
setwd(paste0(path,'/author0'))
pv <- list.files()

for(i in 1: length(pv)){
        
        author<-read.csv(pv[i]) %>% select(-X)
        names(author) <- c('name','org','id','seq')
        fname <- str_replace_all(pv[i],pattern='.csv',replacement = '')
        author <- author %>% mutate(name = str_replace_all(name,pattern = "[0-9]{4}|'|[0-9]+th|[0-9]+nd|[0-9]+st|[0-9]+rd|\\.",
                                                           replacement = ''),
                                    name = str_replace_all(name,pattern = ",|/|'|-|:|\\(|\\)",replacement = ' '),
                                    name = str_replace_all(name,pattern = "[:blank:]{2,}",replacement = ' '),
                                    name = str_replace_all(name,pattern = "^[:blank:]+",replacement = ''),
                                    name = tolower(name),
                                    org = str_replace_all(org,pattern = "[0-9]{4}|'|[0-9]+th|[0-9]+nd|[0-9]+st|[0-9]+rd|\\.",
                                                          replacement = ''),
                                    org = str_replace_all(org,pattern = ",|/|'|-|:|\\(|\\)",replacement = ' '),
                                    org = str_replace_all(org,pattern = "[:blank:]{2,}",replacement = ' '),
                                    org = str_replace_all(org,pattern = "^[:blank:]+",replacement = ''))
        
        auname <- author %>% group_by(name) %>% summarise(count = n()) %>% arrange(desc(count))
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
                cat(j,'\n')
        }
        adjacency_1 <- adjacency[adjacency$merge==1,]
        e <- data.frame(from = adjacency_1$node1,
                        to = adjacency_1$node2)
        
        net <- graph_from_data_frame(e, directed=F, vertices=node)
        
        net_c <- components(net)$membership
        write_json(net_c,path = paste0(path,'/result/',fname,'.json'))
        cat(i,'------------------------------------------------------','\n')
}




