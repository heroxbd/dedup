# caculate likelihood
library(jsonlite)
library(dplyr)
#result_list <- read_json("result_list.json")
likelihood_df <- as.data.frame(1:length(result_list)) 
names(likelihood_df) <- 'step'
likelihood_df$ll_hood <- 0
#similarity <- read.csv('similarity.csv')

names(arcs_org) <- c('node1','node2','similarity')
similarity <- arcs_org

for(i in 1:length(result_list)){
        similarity$edge <- 0
        result <- result_list[[i]]
        ll_hood <- 1
        similarity_ll <- similarity
        similarity_ll$dissimilarity <- 1 - similarity_ll$similarity
        
        similarity_ll$log_sim <- log(similarity_ll$similarity)
        similarity_ll$log_dis <- log(similarity_ll$dissimilarity)
        for(j in unique(result$group)){
                node_cluster <- result$node[result$group==j]
                similarity_ll$edge[similarity_ll$node1 %in% node_cluster &
                                           similarity_ll$node2 %in% node_cluster] <- 1
        }
        
        ll_hood <- sum(similarity_ll$log_sim[similarity_ll$edge==1])+
                sum(similarity_ll$log_dis[similarity_ll$edge==0])
        likelihood_df$ll_hood[i] <- ll_hood
}
# 返回结果

likelihood_df <- likelihood_df %>% arrange(desc(ll_hood)) 
opt <- likelihood_df$step[1]
result_list[[opt]]

plot( x= likelihood_df$step, y = likelihood_df$ll_hood)
