#AP clustering
# install.packages("apcluster")
library(apcluster)
library(dplyr)
path1 <- "/Users/ZLab/Documents/Dropbox/test/validate_val"
path2 <- "/Users/ZLab/Documents/Dropbox/test/id_pairs"

predt_file <- list.files(path1)
pair_file <- list.files(path2)
pair_file <- pair_file[pair_file%in%predt_file ]
au_name <- str_remove(predt_file,pattern='.h5')

ap_assingment <- list()
for(n in 1: length(predt_file)){
        # load data
        similarity <-  cbind(h5read(paste0(path2,'/',pair_file[n]),'id_pairs'),h5read(paste0(path1,'/',predt_file[n]),'/prediction'))
        names(similarity) <- c('node1','node2','similarity')
        node0 <- unique(c(similarity$node1,similarity$node2))
        # patch the node-node sim
        similarity_new <- data.frame(cbind(node0,node0))
        similarity_new$similarity <- 0
        names(similarity_new) <- c('node1','node2','similarity')
        similarity_final <- rbind(similarity_new,similarity)
        similarity_final <- spread(similarity_final,key=node2,value = similarity) 
        rownames(similarity_final) <- similarity_final$node1
        similarity_final$node1 <- NULL
        similarity_mt <- as.matrix(similarity_final)
        ap_cluster <- apcluster(similarity_mt)
        assingment <- list()
        for(i in 1: length(ap_cluster)){
                assingment[[i]] <- names(ap_cluster[[i]])
                cat(i,'\n')
        }
        ap_assingment[[n]] <- assingment
        #names(final_assingment[[n]])
        cat('终于有了第',n,'次结果了，注意看一下')
}
names(ap_assingment) <- au_name
write_json(ap_assingment, path = 'ap_assignment.json')




