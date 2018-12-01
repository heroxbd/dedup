## merge final assignment together 
library(jsonlite)
library(stringr)
rm(list=ls())
#path <- '/Users/ZLab/Downloads/Amine/validate'
path <-'/Users/ZLab/Documents/Dropbox/test/result'
setwd(path)

assignment_file <- list.files(path)
final_assignment <- list()

for(n in 1: length(assignment_file)){
        assignment <- read_json(paste0(path,'/',assignment_file[n]))
        final_assignment[[n]] <- assignment
        names(final_assignment[[n]]) <- c(str_replace_all(assignment_file[n], pattern = 'final_result_|.h5.json', replacement = ''))
        }

write_json(final_assignment,  "final_assignment.json")
