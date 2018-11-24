library(stringr)
# install.packages('argparse')
#require(argparse)
library(dplyr)
setwd("/Users/ZLab/Downloads")
item <- read.csv('./aminer/item_final.csv')
author <- read.csv('./aminer/author_final.csv')
item <- item %>% mutate(venue = str_replace_all(venue,pattern = "[0-9]{4}|'|[0-9]+th|[0-9]+nd|[0-9]+st|[0-9]+rd|\\.",
                                                replacement = ''),
                        venue = str_replace_all(venue,pattern = ",|/|'|-|:|\\(|\\)",replacement = ' '),
                        venue = str_replace_all(venue,pattern = "[:blank:]{2,}",replacement = ' '))
author <- author %>% mutate(name = str_replace_all(name,pattern = "[0-9]{4}|'|[0-9]+th|[0-9]+nd|[0-9]+st|[0-9]+rd|\\.",
                                                    replacement = ''),
                            name = str_replace_all(name,pattern = ",|/|'|-|:|\\(|\\)",replacement = ' '),
                            name = str_replace_all(name,pattern = "[:blank:]{2,}",replacement = ' '),
                            org = str_replace_all(org,pattern = "[0-9]{4}|'|[0-9]+th|[0-9]+nd|[0-9]+st|[0-9]+rd|\\.",
                                                   replacement = ''),
                            org = str_replace_all(org,pattern = ",|/|'|-|:|\\(|\\)",replacement = ' '),
                            org = str_replace_all(org,pattern = "[:blank:]{2,}",replacement = ' ')) 
write.csv(item,file='item.csv')
write.csv(author,file='author.csv')
