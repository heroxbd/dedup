#!/usr/bin/env Rscript
## merge final assignment together 
library(jsonlite)
library(stringr)

require(argparse)
psr <- ArgumentParser(description="from similarity to cluster")
psr$add_argument("ipt", nargs="+", help="pairwise similarity input to merge")
psr$add_argument("-o", dest="opt", help="cluster output")
args <- psr$parse_args()

assignment_file <- args$ipt
final_assignment <- list()

for(n in 1: length(assignment_file)){
    final_assignment[[sub(".json", "", basename(assignment_file[n]))]] <- read_json(assignment_file[n], simplifyVector = TRUE)
}

write_json(final_assignment, args$opt)
