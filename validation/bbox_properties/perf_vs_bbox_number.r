# =============================================================================
# Performance vs Number of Bounding Boxes (binned)
# =============================================================================
# For each scenario, images are sorted into 10 equal-count percentile bins by
# the number of arthropod bounding boxes they contain.  Within each bin the
# mean performance (± bootstrap 95 % CI) is plotted so scenarios can be
# compared visually.
#
# Two plots are produced:
#   • IoU   – mean IoU per image, binned by bbox count
#   • F1    – F1-score per image, binned by bbox count
#
# Run from the project root directory.
# =============================================================================

library(ggplot2)
library(dplyr)
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

scenario_levels <- scenario_names  # display order

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
    per_image_list <- vector("list", length(scenario_names))

    for (i in seq_along(scenario_names)) {
        if (!file.exists(csv_paths[i])) {
            warning(sprintf("CSV not found, skipping: %s", csv_paths[i]))
            next
        }
        df <- read.csv(csv_paths[i], stringsAsFactors = FALSE)
        df$IoUs_with_zeros <- lapply(df$IoUs_with_zeros, parse_list_string)

        per_image_list[[i]] <- data.frame(
            num_bboxes = sapply(df$IoUs_with_zeros, length),
            F1         = df$F1,
            mean_IoU   = sapply(df$IoUs_with_zeros,
                                function(x) if (length(x) > 0) mean(x) else NA_real_),
            scenario   = scenario_names[i],
            stringsAsFactors = FALSE
        )
    }

    do.call(rbind, Filter(Negate(is.null), per_image_list))
}

# ── shared theme ─────────────────────────────────────────────────────────────

bbox_theme <- function() {
    theme_classic() +
    theme(
        text              = element_text(size = 14, color = "#333333"),
        axis.text         = element_text(size = 12, color = "#333333"),
        axis.title        = element_text(size = 14),
        axis.text.x       = element_text(angle = 30, hjust = 1),
        legend.position   = "right",
        legend.title      = element_text(size = 13),
        legend.text       = element_text(size = 11),
        legend.background = element_rect(fill = "#f5f5f5", color = "gray80",
                                         linewidth = 0.4),
        panel.grid.major.y = element_line(color = "grey92", linewidth = 0.3),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.3),
        panel.grid.minor  = element_blank(),
        plot.margin       = margin(12, 12, 12, 12)
    )
}

# ── plotting function ─────────────────────────────────────────────────────────

#' Plot performance vs binned bbox count for multiple scenarios
#'
#' @param data         data frame with columns: x_var, metric_value, scenario
#' @param x_label      x-axis label string
#' @param y_label      y-axis label string
#' @param output_path  file path for the saved PNG
#' @param n_bins       number of equal-count bins (default 10)
plot_binned_perf <- function(data, x_label, y_label, output_path, n_bins = 10) {
    data <- data %>% filter(!is.na(x_var) & !is.na(metric_value))

    # Create decile bins
    data <- data %>% mutate(bin = ntile(x_var, n_bins))

    # X-axis labels: integer range of each bin so low-count bins stay
    # distinguishable even when many images share the same bbox count (e.g. 1).
    bin_labels <- data %>%
        group_by(bin) %>%
        summarise(
            bin_min = as.integer(floor(min(x_var, na.rm = TRUE))),
            bin_max = as.integer(ceiling(max(x_var, na.rm = TRUE))),
            .groups = "drop"
        ) %>%
        arrange(bin) %>%
        mutate(label = ifelse(bin_min == bin_max,
                              as.character(bin_min),
                              paste0(bin_min, "\u2013", bin_max)))

    # Pre-compute mean + 95 % bootstrap CI per bin × scenario.
    # Using explicit geoms with a single shared position_dodge avoids the
    # misalignment that occurs when multiple stat_summary calls each create
    # their own independent dodge state, which caused each scenario line to
    # start one bin late.
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
        scale_x_discrete(labels = bin_labels$label) +
        scale_y_continuous(breaks = seq(0, 1, by = 0.1), limits = c(0, 1)) +
        scale_color_manual(values = scenario_colors, name = "Scenario") +
        labs(x = x_label, y = y_label) +
        bbox_theme()

    dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_path, plot = p, width = 11, height = 6, dpi = 300)
    cat(sprintf("Saved: %s\n", output_path))
    invisible(p)
}

# ── main ─────────────────────────────────────────────────────────────────────

cat("Loading validation data...\n")
data_all <- load_validation_data(scenario_names, csv_paths)

# --- IoU vs bbox count -------------------------------------------------------
cat("\nPlotting IoU vs bbox count...\n")
iou_data <- data_all %>% rename(x_var = num_bboxes, metric_value = mean_IoU)

p_iou <- plot_binned_perf(
    data        = iou_data,
    x_label     = "Number of arthropods per image (percentile bins, mean shown)",
    y_label     = "Mean IoU per image",
    output_path = "validation/bbox_properties/plots/perf_vs_bbox_number_IoU.png"
)

# --- F1 vs bbox count --------------------------------------------------------
cat("Plotting F1 vs bbox count...\n")
f1_data <- data_all %>% rename(x_var = num_bboxes, metric_value = F1)

p_f1 <- plot_binned_perf(
    data        = f1_data,
    x_label     = "Number of arthropods per image (percentile bins, mean shown)",
    y_label     = "F1-score",
    output_path = "validation/bbox_properties/plots/perf_vs_bbox_number_F1.png"
)

# --- Combined stacked export (F1 above IoU) ---------------------------------
combined_path <- "validation/bbox_properties/plots/perf_vs_bbox_number_combined.png"
arr <- arrangeGrob(p_f1, p_iou, ncol = 1)
ggsave(combined_path, plot = arr, width = 11, height = 12, dpi = 300)
cat(sprintf("Saved: %s\n", combined_path))

cat("\nDone.\n")
