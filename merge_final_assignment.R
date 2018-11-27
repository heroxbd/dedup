## merge final assignment together 
library(jsonlite)
library(stringr)
rm(list=ls())
path <- '/Users/ZLab/Downloads/Amine'
assignment_file <- list.files(path)
final_assignment <- list()

for(n in 1: length(assignment_file)){
        assignment <- read_json(paste0(path,'/',assignment_file[n]))
        final_assignment[[n]] <- assignment
        final_assignment[[n]] <- str_replace_all(assignment_file[n], pattern = 'result_|.h5.json', replacement = '')
}