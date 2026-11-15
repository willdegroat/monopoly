args <- commandArgs(trailingOnly = TRUE)

package_version <- function(package) {
    if (!requireNamespace(package, quietly = TRUE)) {
        return(NA_character_)
    }
    as.character(utils::packageVersion(package))
}

if (length(args) == 1 && args[[1]] == '--versions') {
    versions <- c(
        R = paste(R.version$major, R.version$minor, sep = '.'),
        FracFixR = package_version('FracFixR'),
        DESeq2 = package_version('DESeq2'),
        edgeR = package_version('edgeR')
    )
    for (package in names(versions)) {
        cat(package, versions[[package]], sep = '\t')
        cat('\n')
    }
    quit(save = 'no', status = 0)
}

if (length(args) != 5) {
    stop(
        paste(
            'Expected counts, metadata, results, fractions, and methods',
            'arguments.'
        )
    )
}

counts_path <- args[[1]]
metadata_path <- args[[2]]
results_path <- args[[3]]
fractions_path <- args[[4]]
methods <- strsplit(args[[5]], ',', fixed = TRUE)[[1]]
allowed_methods <- c('deseq2', 'edgeR', 'fracfixr')
if (any(!methods %in% allowed_methods) || anyDuplicated(methods)) {
    stop('Methods must be unique members of deseq2, edgeR, and fracfixr.')
}

counts_frame <- utils::read.delim(
    counts_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
)
if (!'geneId' %in% colnames(counts_frame)) {
    stop('Count input must contain geneId.')
}
genes <- counts_frame$geneId
counts <- as.matrix(counts_frame[, setdiff(colnames(counts_frame), 'geneId')])
rownames(counts) <- genes
storage.mode(counts) <- 'integer'
metadata <- utils::read.delim(
    metadata_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
)
required_metadata <- c('sampleId', 'participantId', 'condition', 'fraction')
if (!all(required_metadata %in% colnames(metadata))) {
    stop('Metadata columns do not match the benchmark contract.')
}
if (!identical(colnames(counts), metadata$sampleId)) {
    stop('Count columns and metadata samples must be identically ordered.')
}
if (!identical(unique(metadata$condition), c('Control', 'Test'))) {
    stop('Conditions must be ordered Control, Test.')
}
if (!setequal(unique(metadata$fraction), c('input', 'monosome', 'polysome'))) {
    stop('Fractions must be input, monosome, and polysome.')
}

empty_results <- function(method, estimand, runtime, error) {
    data.frame(
        method = method,
        estimand = estimand,
        geneId = genes,
        estimate = NA_real_,
        standardError = NA_real_,
        pValue = NA_real_,
        adjustedPValue = NA_real_,
        runtimeSeconds = runtime,
        errorMessage = error,
        stringsAsFactors = FALSE
    )
}

build_design <- function(metadata) {
    participant <- factor(
        metadata$participantId,
        levels = unique(metadata$participantId)
    )
    fraction <- factor(
        metadata$fraction,
        levels = c('input', 'monosome', 'polysome')
    )
    condition <- factor(
        metadata$condition,
        levels = c('Control', 'Test')
    )
    design <- stats::model.matrix(
        ~ 0 + participant + fraction + condition:fraction
    )
    input_interaction <- which(
        colnames(design) == 'fractioninput:conditionTest'
    )
    if (length(input_interaction) != 1) {
        stop('Could not identify the redundant input interaction.')
    }
    design <- design[, -input_interaction, drop = FALSE]
    if (qr(design)$rank != ncol(design)) {
        stop('Comparator design is not full rank.')
    }
    contrast <- numeric(ncol(design))
    names(contrast) <- colnames(design)
    contrast[['fractionpolysome:conditionTest']] <- 1
    contrast[['fractionmonosome:conditionTest']] <- -1
    list(matrix = design, contrast = contrast)
}

fraction_size_factors <- function(counts, metadata) {
    factors <- numeric(ncol(counts))
    for (fraction in c('input', 'monosome', 'polysome')) {
        selected <- metadata$fraction == fraction
        factors[selected] <- DESeq2::estimateSizeFactorsForMatrix(
            counts[, selected, drop = FALSE],
            type = 'ratio'
        )
    }
    factors
}

run_deseq2 <- function() {
    started <- proc.time()[['elapsed']]
    tryCatch({
        if (!requireNamespace('DESeq2', quietly = TRUE)) {
            stop('DESeq2 is not installed.')
        }
        specification <- build_design(metadata)
        column_data <- data.frame(row.names = metadata$sampleId)
        dds <- DESeq2::DESeqDataSetFromMatrix(
            countData = counts,
            colData = column_data,
            design = specification$matrix
        )
        BiocGenerics::sizeFactors(dds) <- fraction_size_factors(counts, metadata)
        dds <- DESeq2::DESeq(
            dds,
            fitType = 'parametric',
            minReplicatesForReplace = Inf,
            quiet = TRUE
        )
        observed <- DESeq2::results(
            dds,
            contrast = specification$contrast,
            independentFiltering = FALSE,
            cooksCutoff = FALSE
        )
        runtime <- proc.time()[['elapsed']] - started
        data.frame(
            method = 'deseq2',
            estimand = 'polyVsMono',
            geneId = rownames(observed),
            estimate = observed$log2FoldChange,
            standardError = observed$lfcSE,
            pValue = observed$pvalue,
            adjustedPValue = observed$padj,
            runtimeSeconds = runtime,
            errorMessage = NA_character_,
            stringsAsFactors = FALSE
        )
    }, error = function(error) {
        runtime <- proc.time()[['elapsed']] - started
        empty_results(
            'deseq2',
            'polyVsMono',
            runtime,
            paste(class(error)[[1]], conditionMessage(error), sep = ': ')
        )
    })
}

fraction_tmm_factors <- function(y, metadata) {
    factors <- numeric(ncol(y))
    for (fraction in c('input', 'monosome', 'polysome')) {
        selected <- metadata$fraction == fraction
        subset <- edgeR::DGEList(counts = y$counts[, selected, drop = FALSE])
        subset <- edgeR::calcNormFactors(subset, method = 'TMM')
        factors[selected] <- subset$samples$norm.factors
    }
    factors
}

run_edger <- function() {
    started <- proc.time()[['elapsed']]
    tryCatch({
        if (!requireNamespace('edgeR', quietly = TRUE)) {
            stop('edgeR is not installed.')
        }
        specification <- build_design(metadata)
        y <- edgeR::DGEList(counts = counts)
        y$samples$norm.factors <- fraction_tmm_factors(y, metadata)
        y <- edgeR::estimateDisp(
            y,
            specification$matrix,
            robust = TRUE
        )
        fit <- edgeR::glmQLFit(
            y,
            specification$matrix,
            robust = TRUE
        )
        test <- edgeR::glmQLFTest(fit, contrast = specification$contrast)
        observed <- edgeR::topTags(
            test,
            n = Inf,
            adjust.method = 'BH',
            sort.by = 'none'
        )$table
        observed <- observed[genes, , drop = FALSE]
        runtime <- proc.time()[['elapsed']] - started
        data.frame(
            method = 'edgeR',
            estimand = 'polyVsMono',
            geneId = rownames(observed),
            estimate = observed$logFC,
            standardError = NA_real_,
            pValue = observed$PValue,
            adjustedPValue = observed$FDR,
            runtimeSeconds = runtime,
            errorMessage = NA_character_,
            stringsAsFactors = FALSE
        )
    }, error = function(error) {
        runtime <- proc.time()[['elapsed']] - started
        empty_results(
            'edgeR',
            'polyVsMono',
            runtime,
            paste(class(error)[[1]], conditionMessage(error), sep = ': ')
        )
    })
}

fracfixr_annotation <- function(metadata) {
    type <- c(
        input = 'Total',
        monosome = 'Monosome',
        polysome = 'Polysome'
    )[metadata$fraction]
    data.frame(
        Sample = metadata$sampleId,
        Condition = metadata$condition,
        Type = unname(type),
        Replicate = metadata$participantId,
        stringsAsFactors = FALSE
    )
}

standardize_fracfixr <- function(observed, type, test, runtime) {
    prefix <- if (test == 'GLM') {
        if (type == 'Polysome') {
            'polysomeProportion'
        } else {
            'monosomeProportion'
        }
    } else if (type == 'Polysome') {
        'polysomeVsRemainder'
    } else {
        'monosomeVsRemainder'
    }
    suffix <- if (test == 'GLM') 'Glm' else test
    estimand <- paste0(prefix, suffix)
    aligned <- merge(
        data.frame(geneId = genes, stringsAsFactors = FALSE),
        observed,
        by.x = 'geneId',
        by.y = 'transcript',
        all.x = TRUE,
        sort = FALSE
    )
    aligned <- aligned[match(genes, aligned$geneId), , drop = FALSE]
    error <- if (!any(is.finite(aligned$pval))) {
        paste('FracFixR', test, 'returned no finite p-values.')
    } else {
        NA_character_
    }
    data.frame(
        method = 'fracfixr',
        estimand = estimand,
        geneId = aligned$geneId,
        estimate = aligned$log2FC,
        standardError = NA_real_,
        pValue = aligned$pval,
        adjustedPValue = aligned$padj,
        runtimeSeconds = runtime,
        errorMessage = error,
        stringsAsFactors = FALSE
    )
}

extract_fracfixr_fractions <- function(fixed) {
    observed <- fixed$Fractions
    key <- paste(metadata$condition, metadata$participantId, sep = '_')
    map <- unique(data.frame(
        key = key,
        participantId = metadata$participantId,
        condition = metadata$condition,
        stringsAsFactors = FALSE
    ))
    row_map <- match(observed$Replicate, map$key)
    if (anyNA(row_map)) {
        stop('Could not map FracFixR fraction rows to participants.')
    }
    output <- list()
    for (fraction in c('Monosome', 'Polysome', 'Lost')) {
        output[[fraction]] <- data.frame(
            method = 'fracfixr',
            participantId = map$participantId[row_map],
            condition = map$condition[row_map],
            fraction = tolower(fraction),
            estimatedFraction = observed[[fraction]],
            stringsAsFactors = FALSE
        )
    }
    do.call(rbind, output)
}

run_fracfixr <- function() {
    started <- proc.time()[['elapsed']]
    tryCatch({
        if (!requireNamespace('FracFixR', quietly = TRUE)) {
            stop('FracFixR is not installed.')
        }
        annotation <- fracfixr_annotation(metadata)
        fixed <- FracFixR::FracFixR(
            MatrixCounts = counts,
            Annotation = annotation,
            parallel = FALSE
        )
        observed_results <- list()
        for (test in c('GLM', 'Logit', 'Wald')) {
            for (type in c('Polysome', 'Monosome')) {
                observed <- FracFixR::DiffPropTest(
                    NormObject = fixed,
                    Conditions = c('Control', 'Test'),
                    Types = type,
                    Test = test
                )
                key <- paste(type, test, sep = '_')
                observed_results[[key]] <- list(
                    observed = observed,
                    type = type,
                    test = test
                )
            }
        }
        runtime <- proc.time()[['elapsed']] - started
        outputs <- lapply(
            observed_results,
            function(result) standardize_fracfixr(
                result$observed,
                result$type,
                result$test,
                runtime
            )
        )
        list(
            results = do.call(rbind, outputs),
            fractions = extract_fracfixr_fractions(fixed)
        )
    }, error = function(error) {
        runtime <- proc.time()[['elapsed']] - started
        message <- paste(class(error)[[1]], conditionMessage(error), sep = ': ')
        list(
            results = rbind(
                empty_results(
                    'fracfixr',
                    'polysomeProportionGlm',
                    runtime,
                    message
                ),
                empty_results(
                    'fracfixr',
                    'monosomeProportionGlm',
                    runtime,
                    message
                ),
                empty_results(
                    'fracfixr',
                    'polysomeVsRemainderLogit',
                    runtime,
                    message
                ),
                empty_results(
                    'fracfixr',
                    'monosomeVsRemainderLogit',
                    runtime,
                    message
                ),
                empty_results(
                    'fracfixr',
                    'polysomeVsRemainderWald',
                    runtime,
                    message
                ),
                empty_results(
                    'fracfixr',
                    'monosomeVsRemainderWald',
                    runtime,
                    message
                )
            ),
            fractions = data.frame()
        )
    })
}

result_parts <- list()
fraction_results <- data.frame()
for (method in methods) {
    if (method == 'deseq2') {
        result_parts[[method]] <- run_deseq2()
    } else if (method == 'edgeR') {
        result_parts[[method]] <- run_edger()
    } else if (method == 'fracfixr') {
        observed <- run_fracfixr()
        result_parts[[method]] <- observed$results
        fraction_results <- observed$fractions
    }
}
results <- do.call(rbind, result_parts)
utils::write.table(
    results,
    results_path,
    sep = '\t',
    row.names = FALSE,
    quote = FALSE,
    na = ''
)
utils::write.table(
    fraction_results,
    fractions_path,
    sep = '\t',
    row.names = FALSE,
    quote = FALSE,
    na = ''
)
