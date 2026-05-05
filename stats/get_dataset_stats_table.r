# =============================================================================
# Dataset Statistics Table Generator
# =============================================================================
# Computes per-dataset statistics (# images, # objects, bbox size stats) from
# YOLO-format label directories, exports a CSV, and prints LaTeX table code.
#
# Usage:
#   Option A – compute from raw directories and save CSV:
#     datasets <- list(
#       "ArthroNat" = "dataset",
#       "Flatbug"   = "/media/disk2/flatbug-yolo-split"
#     )
#     get_dataset_stats_table(datasets = datasets, output_csv = "stats/dataset_stats.csv")
#
#   Option B – read existing CSV and (re-)print the LaTeX table:
#     get_dataset_stats_table(input_csv = "stats/dataset_stats.csv")
# =============================================================================

# -----------------------------------------------------------------------------
# Helper: parse all YOLO .txt label files under a dataset root
#   Scans every subfolder of <dataset_path>/labels/ (train, val, test, …)
# -----------------------------------------------------------------------------
analyze_dataset <- function(name, dataset_path) {
    labels_dir <- file.path(dataset_path, "labels")

    if (!dir.exists(labels_dir)) {
        warning(paste("Labels directory not found:", labels_dir))
        return(NULL)
    }

    label_files <- list.files(labels_dir, pattern = "\\.txt$",
                              recursive = TRUE, full.names = TRUE)

    if (length(label_files) == 0) {
        warning(paste("No .txt label files found under:", labels_dir))
        return(NULL)
    }

    n_images  <- length(label_files)
    all_sizes <- numeric(0)

    for (lf in label_files) {
        lines <- readLines(lf, warn = FALSE)
        for (line in lines) {
            parts <- strsplit(trimws(line), "\\s+")[[1]]
            if (length(parts) >= 5) {
                w <- suppressWarnings(as.numeric(parts[4]))
                h <- suppressWarnings(as.numeric(parts[5]))
                if (!is.na(w) && !is.na(h)) {
                    all_sizes <- c(all_sizes, w * h)
                }
            }
        }
    }

    n_objects <- length(all_sizes)

    if (n_objects == 0) {
        return(data.frame(
            Name      = name,
            n_images  = n_images,
            n_objects = 0L,
            min_size  = NA_real_,
            max_size  = NA_real_,
            mean_size = NA_real_,
            std_size  = NA_real_,
            stringsAsFactors = FALSE
        ))
    }

    data.frame(
        Name      = name,
        n_images  = n_images,
        n_objects = n_objects,
        min_size  = min(all_sizes),
        max_size  = max(all_sizes),
        mean_size = mean(all_sizes),
        std_size  = sd(all_sizes),
        stringsAsFactors = FALSE
    )
}

# -----------------------------------------------------------------------------
# Helper: generate LaTeX table code from a stats data frame
# -----------------------------------------------------------------------------
generate_stats_latex_table <- function(df, caption = "Dataset statistics",
                                       label = "tab:dataset_stats") {
    fmt_int  <- function(x) formatC(x, format = "d", big.mark = ",")
    fmt_size <- function(x) ifelse(is.na(x), "--", sprintf("%.4f", x))

    lines <- c(
        "\\begin{table}[H]",
        "\\centering",
        paste0("\\caption{", caption, "}"),
        paste0("\\label{", label, "}"),
        "\\footnotesize",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        paste(
            "\\thead{\\textbf{Name}}",
            "\\thead{\\textbf{\\# images}}",
            "\\thead{\\textbf{\\# objects}}",
            "\\thead{\\textbf{Min size}}",
            "\\thead{\\textbf{Max size}}",
            "\\thead{\\textbf{Mean size}}",
            "\\thead{\\textbf{Std}}",
            sep = " & "
        ) |> paste0(" \\\\"),
        "\\midrule"
    )

    for (i in seq_len(nrow(df))) {
        row <- df[i, ]
        cells <- paste(
            row$Name,
            fmt_int(row$n_images),
            fmt_int(row$n_objects),
            fmt_size(row$min_size),
            fmt_size(row$max_size),
            fmt_size(row$mean_size),
            fmt_size(row$std_size),
            sep = " & "
        )
        row_line <- paste0(cells, " \\\\")
        if (i %% 2 == 0) {
            row_line <- paste0("\\rowcolor{gray!10} ", row_line)
        }
        lines <- c(lines, row_line)
    }

    lines <- c(lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    )

    paste(lines, collapse = "\n")
}

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
#
# Arguments:
#   datasets   – named list mapping display name -> dataset root path
#                e.g. list("ArthroNat" = "dataset", "Flatbug" = "/path/to/fb")
#   output_csv – (optional) path to write the result CSV
#   input_csv  – (optional) path to an existing stats CSV; when provided,
#                datasets is not needed and the function only prints the table
#   caption    – LaTeX caption string
#   label      – LaTeX label string
# -----------------------------------------------------------------------------
get_dataset_stats_table <- function(datasets  = NULL,
                                    output_csv = NULL,
                                    input_csv  = NULL,
                                    caption    = "Dataset statistics",
                                    label      = "tab:dataset_stats") {

    if (!is.null(input_csv)) {
        # ── Option B: read existing CSV, just print LaTeX ──────────────────
        if (!file.exists(input_csv)) {
            stop(paste("Input CSV not found:", input_csv))
        }
        df <- read.csv(input_csv, stringsAsFactors = FALSE)
        cat(generate_stats_latex_table(df, caption, label))
        cat("\n")
        return(invisible(df))
    }

    # ── Option A: compute from directories ────────────────────────────────
    if (is.null(datasets) || length(datasets) == 0) {
        stop("Provide either 'datasets' (named list) or 'input_csv'.")
    }

    results <- list()
    for (name in names(datasets)) {
        cat(sprintf("Analyzing '%s' ...\n", name))
        row <- analyze_dataset(name, datasets[[name]])
        if (!is.null(row)) {
            results[[length(results) + 1]] <- row
        }
    }

    if (length(results) == 0) {
        stop("No valid datasets could be analyzed.")
    }

    df <- do.call(rbind, results)

    # Print summary table to console
    cat("\n--- Dataset Statistics ---\n")
    print(df, row.names = FALSE)

    if (!is.null(output_csv)) {
        dir.create(dirname(output_csv), showWarnings = FALSE, recursive = TRUE)
        write.csv(df, output_csv, row.names = FALSE)
        cat(sprintf("\nStats CSV saved to: %s\n", output_csv))
    }

    # Print LaTeX table
    cat("\n--- LaTeX table code ---\n")
    cat(generate_stats_latex_table(df, caption, label))
    cat("\n")

    return(invisible(df))
}

# =============================================================================
# Example calls (uncomment to run)
# =============================================================================

# # Option A: compute from directories
# datasets <- list(
#     "ArthroNat" = "dataset",
#     "Flatbug"   = "/media/disk2/flatbug-yolo-split"
# )
# get_dataset_stats_table(
#     datasets   = datasets,
#     output_csv = "stats/dataset_stats.csv",
#     caption    = "Summary statistics for each dataset used in this study.",
#     label      = "tab:dataset_stats"
# )

# # Option B: reprint LaTeX from a previously saved CSV
# get_dataset_stats_table(
#     input_csv = "stats/dataset_stats.csv",
#     caption   = "Summary statistics for each dataset used in this study.",
#     label     = "tab:dataset_stats"
# )
