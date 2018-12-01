# caculate likelihood
library(jsonlite)
library(dplyr)
rm(list=ls())
library(rhdf5)



# 需要读入同样的id_paris, 和 similariry还没有写

arcs_org <- id_pairs
arcs_org$similarity <- similarity
names(arcs_org) <- c('node1','node2','similarity')
similarity <- arcs_org
result_list <- read_json(paste0(path0,'/',result_file[n]))

likelihood_df <- as.data.frame(1:length(result_list)) 
names(likelihood_df) <- 'step'
likelihood_df$ll_hood <- 0

likelihood <- function(i){
        similarity$edge <- 0
        node <- result_list[[i]]$node
        group <- result_list[[i]]$group
        result <- data.frame(cbind(node,group),stringsAsFactors = F)
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
        
        return(ll_hood)
}

# 此处需要优化成parallel 
vapply(1:length(result_list), likelihood)
for(k in 1:length(result_list)){
        likelihood_df$ll_hood[k] <- likelihood(k)
         cat(k,'\n')
}
#8:50开始
plot(likelihood_df$ll_hood, likelihood_df$step)
likelihood_df <- likelihood_df %>% arrange(desc(ll_hood)) 
opt <- likelihood_df$step[1]
node <- result_list[[opt]]$node
group <- result_list[[opt]]$group
result <- data.frame(cbind(node,group),stringsAsFactors = F)
final_result <- list()

for(l in 1: length(unique(result$group))){
        final_result[[l]] <- result$node[result$group==unique(result$group)[l]]
}
max <- length(unlist(final_result))
num <- length(final_result)
node0 <- unique(c(similarity$node1,similarity$node2))

if(max < length(node0)){
        
                final_result <- c(final_result,as.list(node0[!node0%in%unlist(final_result)]))
}
# 写出数据没写
write_json(final_result,  paste0('final_',result_file[n]))
        

# require(plyr)





