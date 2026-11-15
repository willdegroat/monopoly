arguments <- commandArgs(trailingOnly = FALSE)
script_argument <- grep('^--file=', arguments, value = TRUE)
if (length(script_argument) != 1L) {
    stop('Unable to locate the benchmark R requirements.')
}
script_directory <- dirname(
    normalizePath(sub('^--file=', '', script_argument))
)
requirements <- read.delim(
    file.path(script_directory, 'requirements.tsv'),
    stringsAsFactors = FALSE
)
required <- setNames(requirements$version, requirements$package)

required_r <- required[['R']]
observed_r <- paste(R.version$major, R.version$minor, sep = '.')
if (observed_r != required_r) {
    stop(
        paste('R version must be', required_r, 'but observed', observed_r)
    )
}

options(repos = c(CRAN = 'https://cloud.r-project.org'))

install_exact_source <- function(package, version, urls) {
    if (requireNamespace(package, quietly = TRUE) &&
        as.character(packageVersion(package)) == version) {
        return(invisible(TRUE))
    }
    for (url in urls) {
        try(
            install.packages(url, repos = NULL, type = 'source'),
            silent = TRUE
        )
        if (requireNamespace(package, quietly = TRUE) &&
            as.character(packageVersion(package)) == version) {
            return(invisible(TRUE))
        }
    }
    invisible(FALSE)
}

cran_packages <- c(
    'aod',
    'dplyr',
    'future',
    'future.apply',
    'ggplot2',
    'matrixStats',
    'nnls',
    'parallelly',
    'tidyr'
)
missing_cran <- cran_packages[
    !vapply(cran_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_cran)) {
    install.packages(missing_cran)
}

fracfixr_version <- required[['FracFixR']]
install_exact_source(
    'FracFixR',
    fracfixr_version,
    c(
        paste0(
            'https://cran.r-project.org/src/contrib/',
            'FracFixR_', fracfixr_version, '.tar.gz'
        ),
        paste0(
            'https://cran.r-project.org/src/contrib/Archive/FracFixR/',
            'FracFixR_', fracfixr_version, '.tar.gz'
        )
    )
)

if (!requireNamespace('BiocManager', quietly = TRUE)) {
    install.packages('BiocManager')
}
if (as.character(BiocManager::version()) != '3.23') {
    BiocManager::install(
        version = '3.23', ask = FALSE, update = FALSE
    )
}
bioc_packages <- c('DESeq2', 'edgeR')
missing_bioc <- bioc_packages[
    !vapply(bioc_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_bioc)) {
    BiocManager::install(
        missing_bioc, ask = FALSE, update = FALSE
    )
}

for (package in bioc_packages) {
    package_version <- required[[package]]
    install_exact_source(
        package,
        package_version,
        c(
            paste0(
                'https://bioconductor.org/packages/3.23/bioc/src/contrib/',
                package, '_', package_version, '.tar.gz'
            ),
            paste0(
                'https://bioconductor.org/packages/3.23/bioc/src/contrib/',
                'Archive/', package, '/', package, '_', package_version,
                '.tar.gz'
            )
        )
    )
}

required <- required[names(required) != 'R']
observed <- vapply(
    names(required),
    function(package) {
        if (!requireNamespace(package, quietly = TRUE)) {
            return(NA_character_)
        }
        as.character(packageVersion(package))
    },
    character(1)
)
if (!identical(observed, required)) {
    stop(
        paste(
            'R package versions do not match the frozen benchmark:',
            paste(names(observed), observed, collapse = ', ')
        )
    )
}
