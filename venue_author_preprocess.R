#!/usr/bin/env Rscript

library(stringr)
library(dplyr)
library(stringr)
library(dplyr)
require(argparse)

psr <- ArgumentParser(description="clean up the strings")
psr$add_argument("ipt", help="input", nargs="+")
psr$add_argument("-o", dest="opt", help="output")
psr$add_argument("--field", default='item', help="field to manipulate")
args <- psr$parse_args()

dv <- read.csv(args$ipt, row.names=1)

if (args$field == 'item') {
    dv <- dv %>% mutate(venue = str_replace_all(venue,pattern = "[0-9]{4}|'|[0-9]+th|[0-9]+nd|[0-9]+st|[0-9]+rd|\\.",
                                             replacement = ''),
                     venue = str_replace_all(venue,pattern = ",|/|'|\"|-|:|\\(|\\)",replacement = ' '),
                     venue = str_replace_all(venue,pattern = "[:blank:]{2,}",replacement = ' '),
                     venue = str_replace_all(venue,pattern = "^[:blank:]+",replacement = ''),
                     title = str_replace_all(title,pattern = "[0-9]{4}|'|[0-9]+th|[0-9]+nd|[0-9]+st|[0-9]+rd|\\.",
                                             replacement = ''),
                     title = str_replace_all(title,pattern = ",|/|'|\"|-|:|\\(|\\)",replacement = ' '),
                     title = str_replace_all(title,pattern = "[:blank:]{2,}",replacement = ' '),
                     title = str_replace_all(title,pattern = "^[:blank:]+",replacement = ''))
} else if (args$field == "author") {
    dv <- dv %>% mutate(name = str_replace_all(name,pattern = "[0-9]{4}|'|[0-9]+th|[0-9]+nd|[0-9]+st|[0-9]+rd|\\.",
                                                    replacement = ''),
                             name = str_replace_all(name,pattern = ",|/|'|\"|-|:|\\(|\\)",replacement = ' '),
                             name = str_replace_all(name,pattern = "[:blank:]{2,}",replacement = ' '),
                             name = str_replace_all(name,pattern = "^[:blank:]+",replacement = ''),
                             name = tolower(name),
                             org = str_replace_all(org,pattern = "[0-9]{4}|'|[0-9]+th|[0-9]+nd|[0-9]+st|[0-9]+rd|\\.",
                                                   replacement = ''),
                             org = str_replace_all(org,pattern = ",|/|'|\"|-|:|\\(|\\)",replacement = ' '),
                             org = str_replace_all(org,pattern = "[:blank:]{2,}",replacement = ' '),
                             org = str_replace_all(org,pattern = "^[:blank:]+",replacement = ''))
}

write.csv(dv,file=args$opt)
