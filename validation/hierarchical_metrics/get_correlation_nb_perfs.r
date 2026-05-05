# Simple lm-based correlation analysis between taxon image count and baseline performance.
#
# This v2 script intentionally keeps the method minimal:
# - one simple linear model per taxonomic level and metric
# - only baseline ArthroNat performance columns are used (e.g., ArthroNat_F1)
# - no Spearman/Pearson or multi-predictor regression

get_strength_label <- function(r_squared) {
    if (is.na(r_squared)) {
        return("unknown")
    }

    if (r_squared < 0.10) {
        return("weak")
    }

    if (r_squared <= 0.30) {
        return("moderate")
    }

    return("strong")
}

get_direction_label <- function(slope) {
    if (is.na(slope)) {
        return("unknown")
    }

    if (slope > 0) {
        return("positive")
    }

    if (slope < 0) {
        return("negative")
    }

    return("flat")
}

safe_model_p_value <- function(fstat, df1, df2) {
    if (is.null(fstat) || is.na(fstat) || is.na(df1) || is.na(df2) || df1 <= 0 || df2 <= 0) {
        return(NA_real_)
    }

    pf(fstat, df1, df2, lower.tail = FALSE)
}

analyze_one_file <- function(csv_path, level, metric) {
    if (!file.exists(csv_path)) {
        message(sprintf("[skip] Missing file: %s", csv_path))
        return(NULL)
    }

    df <- read.csv(csv_path, stringsAsFactors = FALSE)

    if (!"number_of_images" %in% colnames(df)) {
        warning(sprintf("[skip] Missing number_of_images in %s", csv_path))
        return(NULL)
    }

    perf_col <- paste0("ArthroNat_", metric)
    if (!perf_col %in% colnames(df)) {
        warning(sprintf("[skip] Missing baseline column %s in %s", perf_col, csv_path))
        return(NULL)
    }

    model_data <- df[, c("number_of_images", perf_col)]
    model_data <- model_data[complete.cases(model_data), , drop = FALSE]

    n <- nrow(model_data)
    if (n < 3) {
        warning(sprintf("[skip] Not enough rows (n=%d) in %s", n, csv_path))
        return(NULL)
    }

    colnames(model_data) <- c("number_of_images", "baseline_perf")

    lm_model <- lm(number_of_images ~ baseline_perf, data = model_data)
    lm_summary <- summary(lm_model)

    slope <- unname(coef(lm_model)["baseline_perf"])
    intercept <- unname(coef(lm_model)["(Intercept)"])

    fstat <- unname(lm_summary$fstatistic["value"])
    df1 <- unname(lm_summary$fstatistic["numdf"])
    df2 <- unname(lm_summary$fstatistic["dendf"])
    model_p <- safe_model_p_value(fstat, df1, df2)

    data.frame(
        level = level,
        metric = metric,
        input_file = basename(csv_path),
        n = n,
        intercept = intercept,
        slope = slope,
        slope_direction = get_direction_label(slope),
        r_squared = unname(lm_summary$r.squared),
        adjusted_r_squared = unname(lm_summary$adj.r.squared),
        f_statistic = fstat,
        model_p_value = model_p,
        statistically_significant = !is.na(model_p) & model_p < 0.05,
        strength_label = get_strength_label(unname(lm_summary$r.squared)),
        stringsAsFactors = FALSE
    )
}

write_summary_txt <- function(results_df, out_txt) {
    if (nrow(results_df) == 0) {
        writeLines(c("No models were fitted."), out_txt)
        return(invisible(NULL))
    }

    lines <- c(
        "SIMPLE LINEAR CORRELATION ANALYSIS (v2)",
        "=======================================",
        paste("Analysis date:", Sys.time()),
        "Method: lm(number_of_images ~ ArthroNat_metric)",
        "Strength rule: R^2 < 0.10 weak, 0.10-0.30 moderate, > 0.30 strong",
        "Note: slope magnitude depends on metric scale; use R^2 for strength.",
        "",
        "RESULTS:"
    )

    for (i in seq_len(nrow(results_df))) {
        row <- results_df[i, ]
        lines <- c(
            lines,
            sprintf("- %s / %s", toupper(row$level), row$metric),
            sprintf("  n = %d", row$n),
            sprintf("  slope = %.6f (%s)", row$slope, row$slope_direction),
            sprintf("  intercept = %.6f", row$intercept),
            sprintf("  R^2 = %.4f (%s)", row$r_squared, row$strength_label),
            sprintf("  adjusted R^2 = %.4f", row$adjusted_r_squared),
            sprintf("  p-value = %.6g (%s)", row$model_p_value,
                    ifelse(row$statistically_significant, "significant", "not significant")),
            ""
        )
    }

    writeLines(lines, out_txt)
}

run_all_models <- function(
    comparisons_dir = "validation/hierarchical_metrics/comparisons",
    output_dir = "validation/hierarchical_metrics/correlations_v2",
    levels = c("class", "order"),
    metrics = c("F1", "precision", "recall", "mean_IoU")
) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    rows <- list()
    idx <- 1

    for (level in levels) {
        for (metric in metrics) {
            csv_path <- file.path(comparisons_dir, sprintf("comparison_%s_%s.csv", level, metric))
            one <- analyze_one_file(csv_path, level, metric)
            if (!is.null(one)) {
                rows[[idx]] <- one
                idx <- idx + 1
            }
        }
    }

    if (length(rows) == 0) {
        warning("No models were fitted. Check input files and column names.")
        return(invisible(NULL))
    }

    results_df <- do.call(rbind, rows)

    out_csv <- file.path(output_dir, "lm_correlation_summary.csv")
    out_txt <- file.path(output_dir, "lm_correlation_summary.txt")

    write.csv(results_df, out_csv, row.names = FALSE)
    write_summary_txt(results_df, out_txt)

    message(sprintf("[ok] Wrote summary CSV: %s", out_csv))
    message(sprintf("[ok] Wrote summary TXT: %s", out_txt))
    message(sprintf("[ok] Fitted %d model(s)", nrow(results_df)))

    invisible(results_df)
}

if (sys.nframe() == 0) {
    run_all_models()
}
