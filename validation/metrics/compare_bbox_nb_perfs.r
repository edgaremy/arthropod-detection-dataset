# Load necessary libraries
library(ggplot2)  # For plotting
library(dplyr)    # For data manipulation
library(mgcv)     # For GAM (spline) fitting

plot_multiple_scenarios_bbox_metrics <- function(scenario_names, csv_paths, output, metric = 'IoU', test_dataset_name = NULL) {
    # Validate inputs
    if(length(scenario_names) != length(csv_paths)) {
        stop("scenario_names and csv_paths must have the same length")
    }
    
    # Parse string representations of lists into actual lists
    parse_list_string <- function(x) {
        if (is.na(x) || x == "" || x == "[]") {
            return(numeric(0))
        }
        clean_str <- gsub("\\[|\\]", "", x)
        if (clean_str == "") {
            return(numeric(0))
        }
        values <- strsplit(clean_str, ",")[[1]]
        values <- trimws(values)
        return(as.numeric(values))
    }
    
    # Set y-axis label based on metric
    if(metric == 'IoU') {
        ylabel <- 'Average IoU'
    } else if(metric == 'F1') {
        ylabel <- 'F1-score'
    } else {
        stop("metric must be either 'IoU' or 'F1'")
    }
    
    # Process each scenario
    all_plot_data <- list()
    correlation_stats <- list()
    
    for(i in 1:length(scenario_names)) {
        # Read the CSV file
        df <- read.csv(csv_paths[i])
        
        # Convert the string representation of lists to actual lists
        df$IoUs_with_zeros <- lapply(df$IoUs_with_zeros, parse_list_string)
        
        # Calculate number of bboxes per image
        num_bboxes <- sapply(df$IoUs_with_zeros, length)
        
        # Calculate average metric per image
        if(metric == 'IoU') {
            avg_metrics <- sapply(df$IoUs_with_zeros, function(values) {
                if(length(values) > 0) sum(values) / length(values) else 0
            })
        } else if(metric == 'F1') {
            avg_metrics <- df$F1
        }
        
        # Create a data frame for this scenario
        scenario_df <- data.frame(
            num_bboxes = num_bboxes,
            metric_value = avg_metrics,
            scenario = scenario_names[i]
        )
        
        # Count frequency of each point for this scenario
        point_counts <- table(paste(num_bboxes, avg_metrics, sep = "_"))
        scenario_df$count <- as.numeric(point_counts[paste(scenario_df$num_bboxes, scenario_df$metric_value, sep = "_")])
        scenario_df$count[is.na(scenario_df$count)] <- 1
        
        all_plot_data[[i]] <- scenario_df
        
        # Calculate Spearman correlation for this scenario
        valid_idx <- !is.na(avg_metrics)
        X <- num_bboxes[valid_idx]
        y <- avg_metrics[valid_idx]
        cor_test <- cor.test(X, y, method = "spearman")
        correlation_stats[[i]] <- list(
            scenario = scenario_names[i],
            rho = cor_test$estimate,
            p_value = cor_test$p.value
        )
    }
    
    # Combine all data
    plot_df <- do.call(rbind, all_plot_data)
    
    # Use linear fitting
    smooth_method <- "lm"
    smooth_formula <- y ~ x
    fit_name <- "Linear fit"
    
    # Sort scenario names alphabetically for consistent colors
    sorted_scenarios <- sort(scenario_names)
    
    # Use paletteer for ggsci::default_locuszoom palette
    if (!requireNamespace("paletteer", quietly = TRUE)) {
        install.packages("paletteer")
    }
    library(paletteer)
    n_colors <- length(sorted_scenarios)
    colors <- paletteer_d("ggsci::default_locuszoom", n_colors)
    names(colors) <- sorted_scenarios
    
    # Reorder plot_df to match alphabetical scenario order
    plot_df$scenario <- factor(plot_df$scenario, levels = sorted_scenarios)
    
        # Create legend labels with correlation statistics (sorted alphabetically)
    legend_labels <- sapply(sorted_scenarios, function(scenario) {
        # Find the correlation stats for this scenario
        stats_idx <- which(sapply(correlation_stats, function(x) x$scenario == scenario))
        sprintf("%s (ρ=%.2f, p=%.3f)", 
                scenario, 
                correlation_stats[[stats_idx]]$rho, 
                correlation_stats[[stats_idx]]$p_value)
    })
    names(legend_labels) <- sorted_scenarios
    
    # Set legend title based on test dataset name
    if (!is.null(test_dataset_name)) {
        legend_title <- paste("Scenario tested on", test_dataset_name)
    } else {
        legend_title <- "Scenario"
    }
    
    # Create the plot
    p <- ggplot(plot_df, aes(x = num_bboxes, y = metric_value, color = scenario, fill = scenario)) +
        geom_point(aes(size = 20 * log2(count + 1)), alpha = 0.25) +
        geom_smooth(method = smooth_method, formula = smooth_formula, 
                   linewidth = 1.2, se = TRUE, alpha = 0.2) +
        scale_size_continuous(guide = "none") +
        scale_color_manual(values = colors, labels = legend_labels) +
        scale_fill_manual(values = colors, labels = legend_labels) +
        labs(
            x = "Number of arthropods per image",
            y = ylabel,
            color = legend_title,
            fill = legend_title
        ) +
        theme_minimal() +
        theme(
            text = element_text(size = 18, color = "#333333"),
            panel.grid.major = element_line(color = "white", linewidth = 1),
            panel.grid.minor = element_blank(),
            panel.background = element_rect(fill = "#f0f0f0", color = NA),
            plot.background = element_rect(fill = "white", color = NA),
            axis.text = element_text(color = "#333333", size = 14),
            axis.title = element_text(color = "#333333", size = 16),
            axis.ticks = element_blank(),
            axis.line = element_blank(),
            legend.position = "right",
            legend.background = element_rect(fill = "#f0f0f0", color = "gray80", linewidth = 0.5),
            legend.margin = margin(4, 8, 8, 8),
            legend.text = element_text(size = 12),
            legend.title = element_text(size = 14),
            plot.margin = margin(20, 20, 20, 20)
        ) +
        ylim(0, 1)
    
    # Save the plot
    ggsave(output, plot = p, width = 12, height = 8, dpi = 300)
    
    # Return the plot object and correlation statistics
    return(list(plot = p, correlations = correlation_stats))
}

plot_bbox_number_metrics <- function(csv_path, output, metric = 'IoU') {
    # Read the CSV file
    df <- read.csv(csv_path)
    
    # Parse string representations of lists into actual lists
    parse_list_string <- function(x) {
        # Convert string representation like "[1, 2, 3]" to actual vector
        if (is.na(x) || x == "" || x == "[]") {
            return(numeric(0))
        }
        # Remove brackets and split by comma
        clean_str <- gsub("\\[|\\]", "", x)
        if (clean_str == "") {
            return(numeric(0))
        }
        # Split by comma and convert to numeric
        values <- strsplit(clean_str, ",")[[1]]
        values <- trimws(values)  # Remove whitespace
        return(as.numeric(values))
    }
    
    # Convert the string representation of lists to actual lists
    df$IoUs_with_zeros <- lapply(df$IoUs_with_zeros, parse_list_string)
    
    # Set y-axis label based on metric
    if(metric == 'IoU') {
        ylabel <- 'Average IoU'
        y_column <- 'avg_IoUs'
    } else if(metric == 'F1') {
        ylabel <- 'F1-score'
        y_column <- 'F1'
    } else {
        stop("metric must be either 'IoU' or 'F1'")
    }
    
    # Calculate number of bboxes per image
    num_bboxes <- sapply(df$IoUs_with_zeros, length)
    
    # Calculate average metric per image
    if(metric == 'IoU') {
        avg_metrics <- sapply(df$IoUs_with_zeros, function(values) {
            if(length(values) > 0) sum(values) / length(values) else 0
        })
    } else if(metric == 'F1') {
        avg_metrics <- df$F1
    }
    
    # Create a data frame for plotting
    plot_df <- data.frame(
        num_bboxes = num_bboxes,
        metric_value = avg_metrics
    )
    
    # Count frequency of each point
    point_counts <- table(paste(num_bboxes, avg_metrics, sep = "_"))
    plot_df$count <- as.numeric(point_counts[paste(plot_df$num_bboxes, plot_df$metric_value, sep = "_")])
    plot_df$count[is.na(plot_df$count)] <- 1
    
    # Calculate Spearman correlation
    valid_idx <- !is.na(avg_metrics)
    X <- num_bboxes[valid_idx]
    y <- avg_metrics[valid_idx]
    cor_test <- cor.test(X, y, method = "spearman")
    rho <- cor_test$estimate
    p_value <- cor_test$p.value
    
    # Use linear fitting
    smooth_method <- "lm"
    smooth_formula <- y ~ x
    fit_name <- "Linear fit"

    # Create the plot
    p <- ggplot(plot_df, aes(x = num_bboxes, y = metric_value)) +
        geom_point(aes(size = 20 * log2(count + 1), color = "Data points"), alpha = 0.5) +
        geom_smooth(aes(color = fit_name), 
                   method = smooth_method, formula = smooth_formula, 
                   linewidth = 1.2, se = TRUE, fill = "red", alpha = 0.2) +
        geom_point(aes(color = "Statistics"), alpha = 0, size = 0) +  # Invisible point for statistics
        geom_point(aes(color = "95% Confidence Interval"), alpha = 0, size = 0) +  # Invisible point for legend
        scale_size_continuous(guide = "none") +
        labs(
            x = "Number of arthropods per image",
            y = ylabel
        ) +
        scale_color_manual(
            name = "", 
            values = setNames(c("#3b5998", "red", "transparent", "red"), 
                            c("Data points", fit_name, "Statistics", "95% Confidence Interval")),
            labels = setNames(c("Data points", fit_name, 
                              sprintf("Spearman ρ = %.2f\np-value = %.3f", rho, p_value),
                              "95% Confidence Interval"),
                            c("Data points", fit_name, "Statistics", "95% Confidence Interval")),
            breaks = c("Data points", fit_name, "Statistics", "95% Confidence Interval")
        ) +
        guides(
            color = guide_legend(
                override.aes = list(
                    alpha = c(0.5, 1, 0, 0.3),           # Point, line, invisible, fill transparency
                    size = c(4, 0, 0, 5),                # Bigger icons: Point size, no size for line, invisible, fill size
                    linetype = c(0, 1, 0, 0),            # No line, solid line, no line, no line
                    shape = c(16, NA, NA, 15),           # Circle, no shape, no shape, square
                    fill = c(NA, NA, NA, "red")          # No fill, no fill, no fill, red fill
                )
            )
        ) +
        theme_minimal() +
        theme(
            text = element_text(size = 18, color = "#333333"),
            panel.grid.major = element_line(color = "white", linewidth = 1.5),
            panel.grid.minor = element_blank(),
            panel.background = element_rect(fill = "#f0f0f0", color = NA),
            plot.background = element_rect(fill = "white", color = NA),
            axis.text = element_text(color = "#333333", size = 14),
            axis.title = element_text(color = "#333333", size = 16),
            axis.ticks = element_blank(),
            axis.line = element_blank(),
            legend.position = c(0.98, 0.98),     # Top right corner
            legend.justification = c(1, 1),      # Anchor at top right of legend box
            legend.background = element_rect(fill = "#f0f0f0", color = "gray80", linewidth = 0.5),  # Grey background
            legend.margin = margin(4, 8, 8, 8),  # Reduced top margin
            legend.text = element_text(size = 15),  # Bigger font
            legend.key = element_rect(fill = "#f0f0f0", color = NA),  # Grey background for keys
            legend.key.size = unit(1.4, "cm"),     # Bigger icons
            legend.spacing.y = unit(0.2, "cm"),    # Spacing between items
            legend.key.height = unit(1.4, "cm"),   # Explicit height for legend keys
            legend.key.width = unit(1.4, "cm"),    # Explicit width for legend keys
            plot.margin = margin(20, 20, 20, 20)
        ) +
        ylim(0, 1)
    
    # Save the plot
    ggsave(output, plot = p, width = 10, height = 8, dpi = 300)
    
    # Return the plot object
    return(p)
}


# Example: Compare flatbug and mosaicing scenarios for F1 metric on our dataset
scenario_names <- c("ArthroNat mosaic2x2", "flatbug", "ArthroNat+flatbug", "ArthroNat mosaic3x3", "ArthroNat mosaic4x4", "ArthroNat no mosaic")
csv_paths <- c(
    'validation/flatbug/validation_arthro.csv',
    'validation/flatbug/validation_flatbug.csv',
    'validation/flatbug/validation_arthro_and_flatbug.csv',
    'validation/flatbug/validation_arthro_mosaic_33.csv',
    'validation/flatbug/validation_arthro_mosaic_44.csv',
    'validation/flatbug/validation_arthro_nomosaic.csv'
)
result <- plot_multiple_scenarios_bbox_metrics(
    scenario_names = scenario_names,
    csv_paths = csv_paths,
    output = 'validation/flatbug/plots/compare_bbox_nb_F1.png',
    metric = 'F1',
    test_dataset_name = "ArthroNat"
)
result <- plot_multiple_scenarios_bbox_metrics(
    scenario_names = scenario_names,
    csv_paths = csv_paths,
    output = 'validation/flatbug/plots/compare_bbox_nb_IoU.png',
    metric = 'IoU',
    test_dataset_name = "ArthroNat"
)

# Example: Compare flatbug and mosaicing scenarios for F1 metric on flatbug test dataset
scenario_names <- c("ArthroNat mosaic2x2", "flatbug", "ArthroNat+flatbug", "ArthroNat mosaic3x3", "ArthroNat mosaic4x4", "ArthroNat no mosaic")
csv_paths <- c(
    'validation/flatbug/validation(fb)_arthro.csv',
    'validation/flatbug/validation(fb)_flatbug.csv',
    'validation/flatbug/validation(fb)_arthro_and_flatbug.csv',
    'validation/flatbug/validation(fb)_arthro_mosaic_33.csv',
    'validation/flatbug/validation(fb)_arthro_mosaic_44.csv',
    'validation/flatbug/validation(fb)_arthro_nomosaic.csv'
)
result <- plot_multiple_scenarios_bbox_metrics(
    scenario_names = scenario_names,
    csv_paths = csv_paths,
    output = 'validation/flatbug/plots/compare_bbox_nb_F1(fb).png',
    metric = 'F1',
    test_dataset_name = "flatbug"
)
result <- plot_multiple_scenarios_bbox_metrics(
    scenario_names = scenario_names,
    csv_paths = csv_paths,
    output = 'validation/flatbug/plots/compare_bbox_nb_IoU(fb).png',
    metric = 'IoU',
    test_dataset_name = "flatbug"
)

