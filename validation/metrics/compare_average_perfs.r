# Load necessary libraries
library(ggplot2)
library(dplyr)

plot_model_comparison <- function(csv_path, test_dataset, metric = "F1", output_filename = NULL) {
    # Create a bar plot comparing model performances for a specific test dataset.
    # 
    # Args:
    #   csv_path (str): Path to the CSV file containing model comparison data
    #   test_dataset (str): Name of the test dataset to filter by
    #   metric (str): Metric to plot ('F1', 'precision', 'recall', 'mean_IoU')
    #   output_filename (str): Optional full path and filename for saved plot. If NULL, plot is displayed
    # 
    # Returns:
    #   ggplot object: The generated bar plot
    #   
    # Examples:
    #   # Plot F1 scores for arthro test dataset
    #   plot_model_comparison('model_comparison.csv', 'arthro', 'F1')
    #   
    #   # Plot precision for flatbug dataset and save with custom name
    #   plot_model_comparison('model_comparison.csv', 'flatbug', 'precision', 'plots/my_precision_plot.png')
    
    # Validate metric input
    valid_metrics <- c("F1", "precision", "recall", "mean_IoU")
    if (!metric %in% valid_metrics) {
        stop(paste("Invalid metric. Must be one of:", paste(valid_metrics, collapse = ", ")))
    }
    
    # Read the CSV file
    df <- read.csv(csv_path)
    
    # Check if required columns exist
    required_cols <- c("model_name", "test_dataset", "avg_F1", "avg_precision", "avg_recall", "avg_mean_IoU")
    missing_cols <- setdiff(required_cols, colnames(df))
    if (length(missing_cols) > 0) {
        stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
    }
    
    # Filter by test dataset
    filtered_df <- df %>%
        filter(test_dataset == !!test_dataset)
    
    if (nrow(filtered_df) == 0) {
        stop(paste("No data found for test dataset:", test_dataset))
    }
    
    # Select the appropriate column based on metric
    metric_col <- switch(metric,
                        "F1" = "avg_F1",
                        "precision" = "avg_precision", 
                        "recall" = "avg_recall",
                        "mean_IoU" = "avg_mean_IoU")
    
    # Set y-axis label based on metric
    y_label <- switch(metric,
                     "F1" = "Average F1 Score",
                     "precision" = "Average Precision",
                     "recall" = "Average Recall", 
                     "mean_IoU" = "Average Mean IoU")
    
    # Clean model names: replace underscores with spaces
    filtered_df$model_name_clean <- gsub("_", " ", filtered_df$model_name)
    
    # Sort model names alphabetically for consistent colors and ordering
    sorted_models_original <- sort(unique(filtered_df$model_name))
    sorted_models_clean <- gsub("_", " ", sorted_models_original)
    
    # Create color palette - use different colors for each model (alphabetically ordered)
    # Use paletteer for ggsci::default_locuszoom palette
    if (!requireNamespace("paletteer", quietly = TRUE)) {
        install.packages("paletteer")
    }
    library(paletteer)
    n_colors <- length(sorted_models_original)
    colors <- paletteer_d("ggsci::default_locuszoom", n_colors)
    names(colors) <- sorted_models_original

    # Prepare data for plotting with alphabetical ordering by scenario name
    plot_data <- filtered_df %>%
        select(model_name, model_name_clean, value = all_of(metric_col)) %>%
        mutate(model_name_clean = factor(model_name_clean, levels = sorted_models_clean)) %>%
        arrange(model_name_clean)  # Sort alphabetically by scenario name

    if (test_dataset == "arthro") {
        dataset_name = "ArthroNat"
    } else if (test_dataset == "flatbug") {
        dataset_name = test_dataset
    }

    # Create the bar plot
    p <- ggplot(plot_data, aes(x = model_name_clean, y = value, fill = model_name)) +
        geom_bar(stat = "identity", alpha = 0.8, color = "black", linewidth = 0.5) +
        scale_fill_manual(values = colors) +
        labs(
            title = paste("Model Performance Comparison - average", metric),

            subtitle = paste("Test Dataset:", dataset_name),
            x = "Training Scenario",
            y = y_label,
            fill = "Model"
        ) +
        theme_minimal() +
        theme(
            text = element_text(size = 14, color = "#333333"),
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5, margin = margin(b = 10)),
            plot.subtitle = element_text(size = 14, hjust = 0.5, margin = margin(b = 15)),
            axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
            axis.text.y = element_text(size = 12),
            axis.title = element_text(size = 14, face = "bold"),
            legend.position = "none",  # Remove legend since x-axis already shows model names
            panel.grid.major.x = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_rect(fill = "#f8f9fa", color = NA),
            plot.background = element_rect(fill = "white", color = NA),
            plot.margin = margin(20, 20, 20, 20)
        ) +
        coord_cartesian(ylim = c(0, 1)) +  # Assuming metrics are between 0 and 1
        scale_y_continuous(
            breaks = seq(0, 1, 0.2),
            labels = paste0(seq(0, 100, 20), "%")
        )
    
    # Add value labels on top of bars
    p <- p + geom_text(
        aes(label = sprintf("%.3f", value)),
        vjust = -0.5,
        size = 4,
        fontface = "bold",
        color = "#333333"
    )
    
    # Save plot if output filename is provided
    if (!is.null(output_filename)) {
        # Create directory if it doesn't exist
        dir_path <- dirname(output_filename)
        if (!dir.exists(dir_path)) {
            dir.create(dir_path, recursive = TRUE)
        }
        
        ggsave(output_filename, plot = p, width = 10, height = 8, dpi = 300)
        cat(paste("Plot saved to:", output_filename, "\n"))
    }
    
    # Return the plot object
    return(p)
}



# Example usage:

plot_model_comparison('validation/flatbug/model_comparison.csv', 'arthro', 'F1', 'validation/flatbug/plots/arthro_f1_comparison.png')
plot_model_comparison('validation/flatbug/model_comparison.csv', 'flatbug', 'F1', 'validation/flatbug/plots/flatbug_f1_comparison.png')
plot_model_comparison('validation/flatbug/model_comparison.csv', 'arthro', 'mean_IoU', 'validation/flatbug/plots/arthro_mean_iou_comparison.png')
plot_model_comparison('validation/flatbug/model_comparison.csv', 'flatbug', 'mean_IoU', 'validation/flatbug/plots/flatbug_mean_iou_comparison.png')
