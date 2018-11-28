# caculate likelihood
library(jsonlite)
library(dplyr)
rm(list=ls())
library(rhdf5)

path0 <- '/Users/ZLab/Downloads/Amine/mst'
path1 <- '/Users/ZLab/Downloads/Amine/predict'
path2 <- '/Users/ZLab/Downloads/Amine/idpair'

result_file <- list.files(path0)
predt_file <- list.files(path1)
pair_file <- list.files(path2)
pair_file <- pair_file[pair_file%in%predt_file ]

make_decision <- function(n){
        similarity <-  h5read(paste0(path1,'/',predt_file[n]),'/prediction')
        id_pairs <- h5read(paste0(path2,'/',pair_file[n]),'id_pairs')
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
                node <- unlist((result_list[[i]]))[seq(1,length(unlist((result_list[[i]])))-1,2)]
                group <- unlist((result_list[[i]]))[seq(2,length(unlist((result_list[[i]]))),2)]
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
        for(k in 1:length(result_list)){
                likelihood_df$ll_hood[k] <- likelihood(k)
                 cat(k,'\n')
        }
        
        likelihood_df <- likelihood_df %>% arrange(desc(ll_hood)) 
        opt <- likelihood_df$step[1]
        node <- unlist((result_list[[opt]]))[seq(1,length(unlist((result_list[[opt]])))-1,2)]
        group <- unlist((result_list[[opt]]))[seq(2,length(unlist((result_list[[opt]]))),2)]
        result <- data.frame(cbind(node,group),stringsAsFactors = F)
        final_result <- list()
        
        for(l in 1: length(unique(result$group))){
                final_result[[l]] <- result$node[result$group==unique(result$group)[l]]
        }
        max <- length(unlist(final_result))
        num <- length(final_result)
        node0 <- unique(c(similarity$node1,similarity$node2))
        
        if(max < length(node0)){
                final_result[[num+1]] <- node0[!node0%in%unlist(final_result)]
        }
        
        write_json(final_result,  paste0('final_',result_file[n]))
        
}

# require(plyr)

for(n in 1: length(predt_file)){
        make_decision(n)
}





