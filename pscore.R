#!/usr/bin/env Rscript

require(argparse)

psr <- ArgumentParser(description="clean up the strings")
psr$add_argument("ipt", help="input", nargs="+")
psr$add_argument("-o", dest="opt", help="output")
args <- psr$parse_args()

z <- read.csv(args$ipt, col.names=c("name", "score"), header=FALSE)
pdf(args$opt)
hist(z$score, breaks=20, main=sprintf("%s: %.3f +- %.3f", basename(args$ipt), mean(z$score), sd(z$score)), xlab="scores",
     ylab=sprintf("lowest: %s; highest: %s", z[which.min(z$score),'name'], z[which.max(z$score),'name']))
dev.off()
print(summary(z$score))
