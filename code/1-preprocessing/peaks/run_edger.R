# install.packages("readr")
# if (!require("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")

# BiocManager::install("csaw")

# install.packages("tidyverse")

library(edgeR)
library(csaw)
library(Matrix)
library(dplyr)
library(edgeR)
library(tibble)
library(purrr)

args = commandArgs(trailingOnly=TRUE)

folder <- args[1]

print(folder)

x <- t(Matrix::readMM(file.path(folder, "counts.mtx")))

y <- DGEList(as.matrix(x))

var <- readr::read_csv(file.path(folder, "var.csv"))
var$peak <- factor(var$peak)

obs <- readr::read_csv(file.path(folder, "obs.csv"))
obs$cluster <- factor(obs$cluster)
design <- model.matrix(~0+cluster, data=as.data.frame(obs))

clusters <- levels(obs$cluster)

y <- as.matrix(x)
y <- DGEList(y)
model <- estimateDisp(y, design)
# model <- estimateDisp(y, design = design)
fit <- glmFit(model, design)

results <- map_dfr(clusters, function(cluster){
    i <- as.integer(cluster) + 1
    print(i)
    contrast <- rep(-1/(ncol(design)-1), ncol(design))
    contrast[i] <- 1
    print(contrast)

    results <- as_tibble(glmLRT(fit, contrast=contrast)$table)
    results$q_value <- p.adjust(results$PValue, "fdr")
    results$cluster <- cluster
    results$peak <- var$peak
    results
})

write.csv(results, file.path(folder, "results.csv"))

