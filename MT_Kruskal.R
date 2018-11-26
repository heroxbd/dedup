#install.packages('optrees')
#install.packages('jsonlite)

#library(optrees)
library(dplyr)
library(igraph)
library(jsonlite)
arcs_org <- read.csv('similarity.csv')
# #-------------------------------------
# # 测试代码
# #-------------------------------------
# setwd('/Users/birdstone/Downloads/testdata')
# item_final <- read.csv('item_final.csv',stringsAsFactors = F)
# item_fliu <- item_final %>% filter(auid == 'f_liu') 
# similarity <- merge(unique(item_fliu$id),unique(item_fliu$id))
# names(similarity) <- c('node1','node2')
# similarity <- similarity %>% mutate(node1 = as.character(node1),node2 = as.character(node2)) %>% filter(node1 < node2)
# similarity$similarity <- runif(dim(similarity)[1],min=0.1,max=1)
# arcs_org <- similarity
disamid <- read.csv('f_liu.csv',stringsAsFactors = F, header =  F)

# import the disambigious ids and devide them into different ids
#----------------------------------------
#----------------------------------------

disamid <- read.csv('disamid.csv',stringsAsFactors = F)
names(arcs_org) <- c('node1','node2','similarity')
names(disamid) <- c('id','count')
if(dim(disamid)[1]>0){
        disam_list <- list()
        for(i in 1:dim(disamid)[1]){
                n <- disamid$count[i]
                # replicate the similarity matrix
                times_1 <- dim(arcs_org[arcs_org[,1]==disamid$id[i],])[1]
                
                id_1 <- rep(paste0(disamid$id[i],'_',1:n),each = times_1)
                
                arcs_diasam_1 <- do.call('rbind',replicate(n,
                                                           arcs_org[arcs_org[,1]==disamid$id[i],],
                                                           simplify = F))
                arcs_diasam_1$node1 <- id_1
                
                times_2 <- dim(arcs_org[arcs_org[,2]==disamid$id[i],])[1]
                id_2 <- rep(paste0(disamid$id[i],'_',1:n),each = times_2)
                arcs_diasam_2 <- do.call('rbind',replicate(n,
                                                           arcs_org[arcs_org[,2]==disamid$id[i],],
                                                           simplify = F))
                arcs_diasam_2$node2 <- id_2
                
                arcs_org <- rbind(arcs_org[arcs_org[,1]!=disamid$id[i]&arcs_org[,2]!=disamid$id[i],],
                                  arcs_diasam_1,arcs_diasam_2)
                
                df <- data.frame(merge(paste0(disamid$id[i],'_',1:n),paste0(disamid$id[i],'_',1:n),all=TRUE),
                                 stringsAsFactors = F)
                names(df) <- c('node1','node2')
                df <- df %>% mutate(node1 = as.character(node1),
                                    node2 = as.character(node2)) %>% filter(node1 < node2)
                df$similarity <- 0
                arcs_org <- rbind(arcs_org,df)
                disam_list[[i]] <- paste0(disamid$id[i],'_',1:n)
                cat(i,'\n')
        }
        
}

arcs_output <- arcs_org #%>% mutate(edge=0)
names(arcs_output) <- c('node1','node2','similarity')
arcs_output <- arcs_output %>% arrange(node1,node2)
arcs_output$node1 <- factor(arcs_output$node1)

# unify a level to facilitate further 
level <- c(levels(arcs_output$node1),arcs_output$node2[length(arcs_output$node1)])
arcs_output <- arcs_output %>%
        mutate(node1 = factor(arcs_output$node1,levels=level),
               node2 = factor(arcs_output$node2,levels=level),
               node1 = as.numeric(node1),
               node2 = as.numeric(node2))

arcs <- as.matrix(arcs_output)
##-----------------------------------------------------------------------
## 
# 这里要使用重名的变量作为一个list输出，然后定义好default的状态！
# msTreeKruskal_old <- function(nodes, arcs) {
#         
#         # Order arcs by weight
#         arcs <- matrix(arcs[order(arcs[, 3]), ], ncol = 3)
#         
#         # Components
#         components <- matrix(c(nodes, nodes), ncol = 2)
#         
#         # Initialize tree with first arc
#         tree.arcs <- matrix(ncol = 3)[-1, ]
#         
#         stages <- 0  # initialize counter
#         stages.arcs <- c()  # vector to store stage number in wich each arc was added
#         
#         # Start with first arc
#         i <- 1
#         # Repeat until we have |N|-1 arcs
#         while(nrow(tree.arcs) < length(nodes) - 1) {
#                 
#                 # Select arc
#                 min.arc <- arcs[i, ]
#                 
#                 # Check components of the two nodes of selected arc
#                 iComp <- components[components[, 1] == min.arc[1], 2]
#                 jComp <- components[components[, 1] == min.arc[2], 2]
#                 if (iComp != jComp) {
#                         # Add arc to msTree
#                         tree.arcs <- rbind(tree.arcs, min.arc)
#                         # Merge components
#                         components[components[, 2] == jComp, 2] <- iComp
#                 }
#                 
#                 stages <- stages + 1  # counter
#                 # Save in which stage an arc was added to the tree and update
#                 stages.arcs <- c(stages.arcs,
#                                  rep(stages, nrow(tree.arcs) - length(stages.arcs)))
#                 # Continue with next arc
#                 i <- i + 1
#                 
#         }
#         
#         # Column names
#         colnames(tree.arcs) <- c("ept1", "ept2", "weight")
#         # Remove row names
#         rownames(tree.arcs) <- NULL
#         
#         output <- list("tree.nodes" = nodes, "tree.arcs" = tree.arcs,
#                        "stages" = stages, "stages.arcs" = stages.arcs)
#         return(output)
#         
# }
##------------------------------------------------------------
## define new Krukal Function
msTreeKruskal_new <- function(nodes, arcs, disam = NULL,dup=0) {
        # disam是新生成的向量
        if(is.null(disam)){
                output <- msTreeKruskal(nodes, arcs)
        }else{
                arcs <- matrix(arcs[order(arcs[, 3]), ], ncol = 3)
                components <- matrix(c(nodes, nodes), ncol = 2)
                tree.arcs <- matrix(ncol = 3)[-1, ]
                stages <- 0  
                stages.arcs <- c()  
                i <- 1
                while(nrow(tree.arcs) < length(nodes) - 1-dup & i <= dim(arcs)[1]) {
                        min.arc <- arcs[i, ]
                        iComp <- components[components[, 1] == min.arc[1], 2]
                        jComp <- components[components[, 1] == min.arc[2], 2]
                        T_F <- TRUE
                        components_1 <- components 
                        components_1[components_1[, 2] == jComp, 2] <- iComp

                        T_F <- (length(unique(components_1[components_1[, 1] %in% disam, 2]))
                                ==length(disam))
                        if ((iComp != jComp) & T_F) {
                                # Add arc to msTree
                                tree.arcs <- rbind(tree.arcs, min.arc)
                                # Merge components
                                components[components[, 2] == jComp, 2] <- iComp
                        }
                        stages <- stages + 1  # counter
                        # Save in which stage an arc was added to the tree and update
                        stages.arcs <- c(stages.arcs,
                                         rep(stages, nrow(tree.arcs) - length(stages.arcs)))
                        # Continue with next arc
                        i <- i + 1
                        #cat(i,'\n')
                }
                # Column names
                colnames(tree.arcs) <- c("ept1", "ept2", "weight")
                # Remove row names
                rownames(tree.arcs) <- NULL
                output <- list("tree.nodes" = nodes, "tree.arcs" = tree.arcs,
                               "stages" = stages, "stages.arcs" = stages.arcs)   
        }
        # Order arcs by weight
        
        return(output)
}


#arcs <- arcs_org
#transfer similarity to weight
arcs[,3] <- 1-arcs[,3]
nodes <- unique(c(arcs[,1],arcs[,2]))
# 判断使用什么函数
if(dim(disamid)[1]==0){
        KKT <- msTreeKruskal(nodes, arcs)
}else{
        disam <- unlist(disam_list)    
        disam_num <- as.numeric(factor(disam,levels=level))
        KKT <- msTreeKruskal_new(nodes, arcs, disam=disam_num,dup=dim(disamid)[1])
        }

# 
result <- KKT$tree.arcs
result_list <- list()

for(i in 1:dim(result)[1]){
        arcs_result <- arcs_output
        if(i ==1){
                result_node <- data.frame(t(result[1,]))
        }else{
                result_node <- data.frame(result[1:i,])
        }
        node <- unique(c(result_node$ept1,result_node$ept2))
        v <- data.frame(name = node)
        e <- data.frame(from = result_node$ept1,
                        to = result_node$ept2)
        g <- graph_from_data_frame(e, directed=F, vertices=v)
        cluster <- data.frame(cbind(names(components(g)$membership),components(g)$membership),stringsAsFactors = F)
        names(cluster) <- c('node','group')
        cluster <- cluster %>% mutate(node = as.numeric(node),
                                      node = level[node]) 
        result_list[[i]] <- cluster
        #cat(i,'\n')
}
write_json(result_list,  "result_list.json")

