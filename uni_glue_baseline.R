install.packages('igraph')
library(igraph)
setwd("/Users/ZLab/Downloads")
library(dplyr)
library(stringr)
# load data
item_final <- read.csv('./Aminer/item_final.csv')
author_final <- read.csv('./Aminer/author_final.csv')
# preprocessing or not 
item_final <- item_final %>% mutate(venue = str_replace_all(venue,pattern = "[0-9]{4}|'|[0-9]+th|[0-9]+nd|[0-9]+st|[0-9]+rd|\\.",
                                                replacement = ''),
                        venue = str_replace_all(venue,pattern = ",|/|'|-|:|\\(|\\)",replacement = ' '),
                        venue = str_replace_all(venue,pattern = "[:blank:]{2,}",replacement = ' '),
                        venue = str_replace_all(venue,pattern = "^[:blank:]+",replacement = ''))

author_final <- author_final %>% mutate(name = str_replace_all(name,pattern = "[0-9]{4}|'|[0-9]+th|[0-9]+nd|[0-9]+st|[0-9]+rd|\\.",
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

# using venue, title, and org as naive groups
# get the 
#key <- item_final$auid[i]
#temp <- item_final[item_final$auid==key,]
temp <- temp %>% mutate(group1 = as.numeric(as.factor(as.character(temp$venue))),
                        group2 = as.numeric(as.factor(as.character(temp$title)))) %>%
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
   
# net2  
index.df <- temp3 %>% group_by(group2) %>% summarise(count=n()) %>% filter(count>1)
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

# union net1-3 and get the final net 
net <- net1 %u% net2 %u% net3
net_c <- components(net)$membership
temp3$group_result <- net_c
# output
write.csv(temp3[c('id','group_result')],file = 'glue_result.csv')
