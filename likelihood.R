#!/usr/bin/env Rscript
# caculate likelihood
library(jsonlite)
require(plyr)
library(dplyr)
library(rhdf5)
require(doMC)
registerDoMC()

# 需要读入同样的id_paris, 和 similariry还没有写
require(argparse)
psr <- ArgumentParser(description="from similarity to cluster")
psr$add_argument("ipt", help="pairwise similarity input")
psr$add_argument("-o", dest="opt", help="cluster output")
psr$add_argument("--id", help="id pairs input")
psr$add_argument("--kruskal", help="kruskal ouptut")
args <- psr$parse_args()

similarity <-  h5read(args$ipt,'/prediction')
arcs_org <- h5read(args$id,'id_pairs')
arcs_org$similarity <- similarity
names(arcs_org) <- c('node1','node2','similarity')
similarity <- arcs_org

result_list <- read_json(args$kruskal)

likelihood_df <- as.data.frame(1:length(result_list))
names(likelihood_df) <- 'step'

likelihood <- function(i){
    similarity$edge <- 0
    node <- unlist((result_list[[i]]))[seq(1,length(unlist((result_list[[i]])))-1,2)]
    group <- unlist((result_list[[i]]))[seq(2,length(unlist((result_list[[i]]))),2)]
    ## node <- result_list[[i]]$node
    ## group <- result_list[[i]]$group
    result <- data.frame(node,group)
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
    
    sum(similarity_ll$log_sim[similarity_ll$edge==1])+sum(similarity_ll$log_dis[similarity_ll$edge==0])
}

# 此处需要优化成parallel 

# likelihood_df$ll_hood <- laply(1:length(result_list), likelihood)

# 改写成二分法，目标是找到第一个差分下降的点，这个点就是我们的最优解
N <- length(result_list)
n <- 1
while(N>n+1){
        m <- floor((N+n)/2)
        diff <- likelihood(m) - likelihood(m-1)
        if(diff>0){
                n <- m
        }else(N <- m)
}

#8:50开始
# pdf(sub('.json', '.pdf', args$opt))
# plot(likelihood_df$ll_hood, type="l", xlab="step", ylab="loglik", 
#      main=sprintf("%s: %s steps", basename(args$opt), length(result_list)))
# dev.off()
# 
# likelihood_df <- likelihood_df %>% arrange(desc(ll_hood)) 

# opt <- likelihood_df$step[1]
opt <- n
node <- unlist((result_list[[opt]]))[seq(1,length(unlist((result_list[[opt]])))-1,2)]
group <- unlist((result_list[[opt]]))[seq(2,length(unlist((result_list[[opt]]))),2)]
result <- data.frame(node,group)
final_result <- list()

for(l in 1:length(unique(result$group))){
        final_result[[l]] <- result$node[result$group==unique(result$group)[l]]
}

max <- length(unlist(final_result))
num <- length(final_result)
node0 <- unique(c(similarity$node1,similarity$node2))

if(max < length(node0)){
    final_result <- c(final_result,as.list(node0[!node0%in%unlist(final_result)]))
}

# 写出数据没写
write_json(final_result, args$opt)
