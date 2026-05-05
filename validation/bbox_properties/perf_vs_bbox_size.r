# =============================================================================
# Performance vs Bounding-Box Size (binned)
# =============================================================================
# For each scenario, arthropod bboxes are sorted into 10 equal-count percentile
# bins by relative size (width × height).  Within each bin the mean performance
# (± bootstrap 95 % CI) is plotted so scenarios can be compared visually.
#
# Two plots are produced per call:
#   • IoU   – one data point per bbox  (unpacked from IoUs_with_zeros / bbox_sizes)
#   • F1    – one data point per image (F1 column, binned by mean image bbox size)
#
# Run from the project root directory.
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(Hmisc)      # for mean_cl_boot
library(gridExtra)

# ── helpers ──────────────────────────────────────────────────────────────────

parse_list_string <- function(x) {
    if (is.na(x) || x == "" || x == "[]") return(numeric(0))
    clean_str <- gsub("\\[|\\]", "", x)
    if (clean_str == "") return(numeric(0))
    as.numeric(trimws(strsplit(clean_str, ",")[[1]]))
}

# ── scenario configuration ────────────────────────────────────────────────────

scenario_names <- c(
    "ArthroNat no mosaic",
    "ArthroNat mosaic2x2",
    "ArthroNat mosaic3x3",
    "ArthroNat mosaic4x4",
    "ArthroNat+flatbug",
    "flatbug"
)

csv_paths <- c(
    "validation/metrics/validation_arthro_nomosaic.csv",
    "validation/metrics/validation_arthro.csv",
    "validation/metrics/validation_arthro_mosaic_33.csv",
    "validation/metrics/validation_arthro_mosaic_44.csv",
    "validation/metrics/validation_arthro_and_flatbug.csv",
    "validation/metrics/validation_flatbug.csv"
)

scenario_levels <- scenario_names  # display order (also alphabetically close)

scenario_colors <- c(
    "ArthroNat no mosaic"         = "#bdd7e7",
    "ArthroNat mosaic2x2"         = "#6baed6",
    "ArthroNat mosaic3x3"         = "#3182bd",
    "ArthroNat mosaic4x4"         = "#08519c",
    # "ArthroNat mosaic6x6"         = "#001f5b",
    "ArthroNat+flatbug"           = "#31a354",
    # "ArthroNat+flatbug mosaic4x4" = "#006d2c",
    "flatbug"                     = "#e6550d"
)

# ── data loading ──────────────────────────────────────────────────────────────

load_validation_data <- function(scenario_names, csv_paths) {
    per_bbox_list  <- vector("list", length(scenario_names))
    per_image_list <- vector("list", length(scenario_names))

    for (i in seq_along(scenario_names)) {
        if (!file.exists(csv_paths[i])) {
            warning(sprintf("CSV not found, skipping: %s", csv_paths[i]))
            next
        }
        df <- read.csv(csv_paths[i], stringsAsFactors = FALSE)
        df$IoUs_with_zeros <- lapply(df$IoUs_with_zeros, parse_list_string)
        df$bbox_sizes       <- lapply(df$bbox_sizes,       parse_list_string)

        # per-image data: mean bbox size + F1 + mean IoU
        per_image_list[[i]] <- data.frame(
            mean_bbox_size = sapply(df$bbox_sizes,
                                    function(x) if (length(x) > 0) mean(x) else NA_real_),
            F1             = df$F1,
            mean_IoU       = sapply(df$IoUs_with_zeros,
                                    function(x) if (length(x) > 0) mean(x) else NA_real_),
            scenario       = scenario_names[i],
            stringsAsFactors = FALSE
        )

        # per-bbox data: individual IoU paired with individual bbox size
        df_long <- df %>%
            mutate(image_id = row_number()) %>%
            unnest(cols = c(IoUs_with_zeros, bbox_sizes)) %>%
            rename(IoU = IoUs_with_zeros, bbox_size = bbox_sizes) %>%
            mutate(scenario = scenario_names[i]) %>%
            select(image_id, IoU, bbox_size, scenario)

        per_bbox_list[[i]] <- df_long
    }

    list(
        per_bbox  = do.call(rbind, Filter(Negate(is.null), per_bbox_list)),
        per_image = do.call(rbind, Filter(Negate(is.null), per_image_list))
    )
}

# ── shared theme ─────────────────────────────────────────────────────────────

bbox_theme <- function() {
    theme_classic() +
    theme(
        text             = element_text(size = 14, color = "#333333"),
        axis.text        = element_text(size = 12, color = "#333333"),
        axis.title       = element_text(size = 14),
        axis.text.x      = element_text(angle = 30, hjust = 1),
        legend.position  = "right",
        legend.title     = element_text(size = 13),
        legend.text      = element_text(size = 11),
        legend.background = element_rect(fill = "#f5f5f5", color = "gray80",
                                         linewidth = 0.4),
        panel.grid.major.y = element_line(color = "grey92", linewidth = 0.3),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.3),
        panel.grid.minor = element_blank(),
        plot.margin      = margin(12, 12, 12, 12)
    )
}

# ── plotting function ─────────────────────────────────────────────────────────

#' Plot performance vs binned bbox size for multiple scenarios
#'
#' @param data         data frame with columns: x_var, metric_value, scenario
#' @param x_label      x-axis label string
#' @param y_label      y-axis label string
#' @param output_path  file path for the saved PNG
#' @param subplot_label subfigure label shown above the plot (e.g., "(a) F1-score")
#' @param n_bins       number of equal-count bins (default 10)
plot_binned_perf <- function(data, x_label, y_label, output_path,
                             subplot_label = NULL, n_bins = 10) {
    data <- data %>% filter(!is.na(x_var) & !is.na(metric_value))

    # Create decile bins
    data <- data %>% mutate(bin = ntile(x_var, n_bins))

    # Per-bin x-axis labels (mean of continuous values, 4 d.p.)
    bin_labels <- data %>%
        group_by(bin) %>%
        summarise(bin_mean = round(mean(x_var, na.rm = TRUE), 4), .groups = "drop") %>%
        arrange(bin)

    # Pre-compute mean + 95 % bootstrap CI per bin × scenario.
    # Using explicit geoms with a single shared position_dodge avoids the
    # misalignment that occurs when multiple stat_summary calls each create
    # their own independent dodge state.
    summary_data <- data %>%
        group_by(bin, scenario) %>%
        summarise(
            mean_val = mean(metric_value, na.rm = TRUE),
            lower    = smean.cl.boot(metric_value)[["Lower"]],
            upper    = smean.cl.boot(metric_value)[["Upper"]],
            .groups  = "drop"
        )

    summary_data$scenario <- factor(summary_data$scenario, levels = scenario_levels)

    pd <- position_dodge(width = 0.5)

    p <- ggplot(summary_data, aes(x = factor(bin), y = mean_val,
                                  colour = scenario, group = scenario)) +
        geom_line(position = pd, linewidth = 0.7, alpha = 0.6) +
        geom_errorbar(aes(ymin = lower, ymax = upper),
                      width = 0.4, position = pd, linewidth = 0.7) +
        geom_point(position = pd, size = 2.5) +
        scale_x_discrete(labels = bin_labels$bin_mean) +
        scale_y_continuous(breaks = seq(0, 1, by = 0.1), limits = c(0, 1)) +
        scale_color_manual(values = scenario_colors, name = "Scenario") +
        labs(x = x_label, y = y_label, title = subplot_label) +
        bbox_theme() +
        theme(
            plot.title = element_text(size = 16, hjust = 0.5, margin = margin(b = 10))
        )

    dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_path, plot = p, width = 11, height = 6, dpi = 300)
    cat(sprintf("Saved: %s\n", output_path))
    invisible(p)
}

# ── main ─────────────────────────────────────────────────────────────────────

cat("Loading validation data...\n")
data_all <- load_validation_data(scenario_names, csv_paths)

# --- IoU vs bbox size (per-bbox data) ----------------------------------------
cat("\nPlotting IoU vs bbox size...\n")
per_bbox_iou <- data_all$per_bbox %>%
    rename(x_var = bbox_size, metric_value = IoU)

plot_binned_perf(
    data        = per_bbox_iou,
    x_label     = "Arthropod apparent size (percentile bins)",
    y_label     = "IoU",
    output_path = "validation/bbox_properties/plots/perf_vs_bbox_size_IoU.png",
    subplot_label = "(b) IoU"
)

# --- F1 vs mean bbox size per image ------------------------------------------
cat("Plotting F1 vs bbox size...\n")
per_image_f1 <- data_all$per_image %>%
    rename(x_var = mean_bbox_size, metric_value = F1)

plot_binned_perf(
    data        = per_image_f1,
    x_label     = "Mean arthropod apparent size per image (percentile bins)",
    y_label     = "F1-score",
    output_path = "validation/bbox_properties/plots/perf_vs_bbox_size_F1.png",
    subplot_label = "(a) F1-score"
)

# --- Combined stacked export (F1 above IoU) ---------------------------------
p_iou <- plot_binned_perf(
    data        = per_bbox_iou,
    x_label     = "Arthropod apparent size (percentile bins)",
    y_label     = "IoU",
    output_path = "validation/bbox_properties/plots/perf_vs_bbox_size_IoU.png",
    subplot_label = "(b) IoU"
)

p_f1 <- plot_binned_perf(
    data        = per_image_f1,
    x_label     = "Mean arthropod apparent size per image (percentile bins)",
    y_label     = "F1-score",
    output_path = "validation/bbox_properties/plots/perf_vs_bbox_size_F1.png",
    subplot_label = "(a) F1-score"
)

combined_path <- "validation/bbox_properties/plots/perf_vs_bbox_size_combined.png"
arr <- arrangeGrob(p_f1, p_iou, ncol = 1)
ggsave(combined_path, plot = arr, width = 11, height = 12, dpi = 300)
cat(sprintf("Saved: %s\n", combined_path))

cat("\nDone.\n")
