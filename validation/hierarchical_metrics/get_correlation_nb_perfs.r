# Load necessary libraries
library(stats)

compute_scenario_correlations <- function(csv_path, output_txt, metric_name) {
    # Compute correlations between image count and all scenario performances for a given metric.
    # 
    # Args:
    #   csv_path (str): Path to the hierarchical metrics CSV file (1 CSV = 1 metric)
    #   output_txt (str): Path to save the correlation results
    #   metric_name (str): Name of the metric for display purposes ('F1', 'precision', 'recall', 'mean_IoU')
    
    # Read the CSV file
    if (!file.exists(csv_path)) {
        stop(paste("CSV file not found:", csv_path))
    }
    
    df <- read.csv(csv_path, stringsAsFactors = FALSE)
    
    # Check required columns
    if (!"number_of_images" %in% colnames(df)) {
        stop("Missing required column: number_of_images")
    }
    
    # Get taxonomic level from column names (first column should be the taxonomic level)
    taxonomic_level <- colnames(df)[1]
    
    # Identify scenario columns (exclude taxonomic level, number_of_images, and summary stats)
    excluded_cols <- c(taxonomic_level, "number_of_images", "mean", "std", "min", "max", "range", "best_scenario", "best_value")
    scenario_cols <- setdiff(colnames(df), excluded_cols)
    
    if (length(scenario_cols) == 0) {
        stop("No scenario columns found in the CSV")
    }
    
    # Remove any rows with NA values for number_of_images
    clean_data <- df[!is.na(df$number_of_images), ]
    
    if (nrow(clean_data) < 3) {
        stop(paste("Not enough data points for analysis. Need at least 3, got", nrow(clean_data)))
    }
    
    # Compute summary statistics for image counts
    n_categories <- nrow(clean_data)
    total_images <- sum(clean_data$number_of_images, na.rm = TRUE)
    mean_count <- mean(clean_data$number_of_images, na.rm = TRUE)
    median_count <- median(clean_data$number_of_images, na.rm = TRUE)
    min_count <- min(clean_data$number_of_images, na.rm = TRUE)
    max_count <- max(clean_data$number_of_images, na.rm = TRUE)
    
    # ===== MULTIPLE CORRELATION ANALYSIS =====
    # Remove rows with any NA values in scenario columns for complete case analysis
    complete_data <- clean_data[complete.cases(clean_data[, c("number_of_images", scenario_cols)]), ]
    
    if (nrow(complete_data) < length(scenario_cols) + 2) {
        stop(paste("Not enough complete cases for multiple correlation analysis. Need at least", 
                  length(scenario_cols) + 2, "got", nrow(complete_data)))
    }
    
    # Prepare data for multiple correlation
    image_count <- complete_data$number_of_images
    scenario_matrix <- as.matrix(complete_data[, scenario_cols])
    
    # ===== PEARSON MULTIPLE CORRELATION (Linear relationships) =====
    if (length(scenario_cols) == 1) {
        # Simple correlation case (only one scenario)
        pearson_R <- abs(cor(image_count, scenario_matrix[, 1], method = "pearson"))
        pearson_R_squared <- pearson_R^2
        pearson_adj_R_squared <- pearson_R_squared
        pearson_f_statistic <- NA
        pearson_p_value <- cor.test(image_count, scenario_matrix[, 1], method = "pearson")$p.value
    } else {
        # Multiple regression to calculate Pearson R
        reg_formula <- as.formula(paste("image_count ~", paste(scenario_cols, collapse = " + ")))
        reg_data <- cbind(image_count = image_count, as.data.frame(scenario_matrix))
        
        tryCatch({
            lm_model <- lm(reg_formula, data = reg_data)
            model_summary <- summary(lm_model)
            
            pearson_R <- sqrt(model_summary$r.squared)
            pearson_R_squared <- model_summary$r.squared
            pearson_adj_R_squared <- model_summary$adj.r.squared
            pearson_f_statistic <- model_summary$fstatistic[1]
            pearson_p_value <- pf(pearson_f_statistic, 
                         model_summary$fstatistic[2], 
                         model_summary$fstatistic[3], 
                         lower.tail = FALSE)
            
            # Individual contributions (standardized coefficients)
            pearson_contributions <- model_summary$coefficients[-1, 1]  # exclude intercept
            pearson_scenario_p_values <- model_summary$coefficients[-1, 4]  # p-values for each scenario
            
        }, error = function(e) {
            stop(paste("Pearson multiple regression failed:", e$message))
        })
    }
    
    # ===== SPEARMAN MULTIPLE CORRELATION (Monotonic relationships) =====
    # Convert all data to ranks for Spearman analysis
    image_count_ranks <- rank(image_count)
    scenario_ranks <- apply(scenario_matrix, 2, rank)
    
    if (length(scenario_cols) == 1) {
        # Simple Spearman correlation case
        spearman_R <- abs(cor(image_count_ranks, scenario_ranks[, 1], method = "pearson"))  # Pearson on ranks = Spearman
        spearman_R_squared <- spearman_R^2
        spearman_adj_R_squared <- spearman_R_squared
        spearman_f_statistic <- NA
        spearman_p_value <- cor.test(image_count, scenario_matrix[, 1], method = "spearman")$p.value
    } else {
        # Multiple regression on ranks (Spearman approach)
        spearman_reg_data <- cbind(image_count_ranks = image_count_ranks, as.data.frame(scenario_ranks))
        spearman_formula <- as.formula(paste("image_count_ranks ~", paste(colnames(scenario_ranks), collapse = " + ")))
        
        tryCatch({
            spearman_lm_model <- lm(spearman_formula, data = spearman_reg_data)
            spearman_model_summary <- summary(spearman_lm_model)
            
            spearman_R <- sqrt(spearman_model_summary$r.squared)
            spearman_R_squared <- spearman_model_summary$r.squared
            spearman_adj_R_squared <- spearman_model_summary$adj.r.squared
            spearman_f_statistic <- spearman_model_summary$fstatistic[1]
            spearman_p_value <- pf(spearman_f_statistic, 
                         spearman_model_summary$fstatistic[2], 
                         spearman_model_summary$fstatistic[3], 
                         lower.tail = FALSE)
            
            # Individual contributions for Spearman (on ranks)
            spearman_contributions <- spearman_model_summary$coefficients[-1, 1]  # exclude intercept
            spearman_scenario_p_values <- spearman_model_summary$coefficients[-1, 4]  # p-values
            
        }, error = function(e) {
            stop(paste("Spearman multiple regression failed:", e$message))
        })
    }
    
    # Store results for both methods
    multiple_correlation_results <- list(
        # Pearson results (linear relationships)
        pearson_R = pearson_R,
        pearson_R_squared = pearson_R_squared,
        pearson_adjusted_R_squared = pearson_adj_R_squared,
        pearson_f_statistic = pearson_f_statistic,
        pearson_p_value = pearson_p_value,
        
        # Spearman results (monotonic relationships)
        spearman_R = spearman_R,
        spearman_R_squared = spearman_R_squared,
        spearman_adjusted_R_squared = spearman_adj_R_squared,
        spearman_f_statistic = spearman_f_statistic,
        spearman_p_value = spearman_p_value,
        
        # General info
        n_complete_cases = nrow(complete_data),
        n_scenarios = length(scenario_cols),
        scenario_names = scenario_cols,
        
        # Method comparison
        R_difference = abs(pearson_R - spearman_R),
        method_agreement = ifelse(abs(pearson_R - spearman_R) < 0.1, "HIGH", 
                                ifelse(abs(pearson_R - spearman_R) < 0.2, "MODERATE", "LOW"))
    )
    
    # Add individual contributions if available
    if (exists("pearson_contributions")) {
        multiple_correlation_results$pearson_contributions = pearson_contributions
        multiple_correlation_results$pearson_scenario_p_values = pearson_scenario_p_values
    }
    
    if (exists("spearman_contributions")) {
        multiple_correlation_results$spearman_contributions = spearman_contributions
        multiple_correlation_results$spearman_scenario_p_values = spearman_scenario_p_values
    }
    
    # Print results to console
    cat("\n")
    cat(paste(rep("=", 80), collapse = ""), "\n")
    cat("CORRELATION ANALYSIS:", toupper(metric_name), "vs IMAGE COUNT\n")
    cat(paste(rep("=", 80), collapse = ""), "\n")
    cat("Input file:", basename(csv_path), "\n")
    cat("Taxonomic level:", toupper(taxonomic_level), "\n")
    cat("Metric analyzed:", toupper(metric_name), "\n")
    cat("Scenarios found:", length(scenario_cols), "\n")
    cat("\n")
    
    cat("DATASET SUMMARY:\n")
    cat(sprintf("  Categories analyzed: %d\n", n_categories))
    cat(sprintf("  Total images: %d\n", total_images))
    cat(sprintf("  Image count - Mean: %.2f, Median: %.0f, Range: %d-%d\n", 
               mean_count, median_count, min_count, max_count))
    cat("\n")
    
    cat("MULTIPLE CORRELATION ANALYSIS:\n")
    cat(paste(rep("=", 80), collapse = ""), "\n")
    
    cat(sprintf("Complete cases used: %d (out of %d categories)\n", 
               multiple_correlation_results$n_complete_cases, n_categories))
    cat(sprintf("Number of scenarios: %d\n", multiple_correlation_results$n_scenarios))
    cat("\n")
    
    # ===== PEARSON RESULTS (Linear relationships) =====
    pearson_strength <- multiple_correlation_results$pearson_R
    if (pearson_strength < 0.3) {
        p_strength <- "WEAK"
    } else if (pearson_strength < 0.7) {
        p_strength <- "MODERATE" 
    } else {
        p_strength <- "STRONG"
    }
    
    cat("PEARSON MULTIPLE CORRELATION (Linear Relationships):\n")
    cat(sprintf("  Multiple R:           %6.3f (%s correlation)\n", 
               multiple_correlation_results$pearson_R, p_strength))
    cat(sprintf("  R-squared:            %6.3f (%.1f%% of variance explained)\n", 
               multiple_correlation_results$pearson_R_squared, multiple_correlation_results$pearson_R_squared * 100))
    
    if (!is.na(multiple_correlation_results$pearson_adjusted_R_squared)) {
        cat(sprintf("  Adjusted R-squared:   %6.3f (%.1f%% adjusted for complexity)\n", 
                   multiple_correlation_results$pearson_adjusted_R_squared, 
                   multiple_correlation_results$pearson_adjusted_R_squared * 100))
    }
    
    if (!is.na(multiple_correlation_results$pearson_f_statistic)) {
        cat(sprintf("  F-statistic:          %6.2f\n", multiple_correlation_results$pearson_f_statistic))
    }
    
    cat(sprintf("  p-value:              %6.4f %s\n", 
               multiple_correlation_results$pearson_p_value,
               ifelse(multiple_correlation_results$pearson_p_value < 0.05, "*", "")))
    cat("\n")
    
    # ===== SPEARMAN RESULTS (Monotonic relationships) =====
    spearman_strength <- multiple_correlation_results$spearman_R
    if (spearman_strength < 0.3) {
        s_strength <- "WEAK"
    } else if (spearman_strength < 0.7) {
        s_strength <- "MODERATE" 
    } else {
        s_strength <- "STRONG"
    }
    
    cat("SPEARMAN MULTIPLE CORRELATION (Monotonic Relationships):\n")
    cat(sprintf("  Multiple R:           %6.3f (%s correlation)\n", 
               multiple_correlation_results$spearman_R, s_strength))
    cat(sprintf("  R-squared:            %6.3f (%.1f%% of variance explained)\n", 
               multiple_correlation_results$spearman_R_squared, multiple_correlation_results$spearman_R_squared * 100))
    
    if (!is.na(multiple_correlation_results$spearman_adjusted_R_squared)) {
        cat(sprintf("  Adjusted R-squared:   %6.3f (%.1f%% adjusted for complexity)\n", 
                   multiple_correlation_results$spearman_adjusted_R_squared, 
                   multiple_correlation_results$spearman_adjusted_R_squared * 100))
    }
    
    if (!is.na(multiple_correlation_results$spearman_f_statistic)) {
        cat(sprintf("  F-statistic:          %6.2f\n", multiple_correlation_results$spearman_f_statistic))
    }
    
    cat(sprintf("  p-value:              %6.4f %s\n", 
               multiple_correlation_results$spearman_p_value,
               ifelse(multiple_correlation_results$spearman_p_value < 0.05, "*", "")))
    cat("\n")
    
    # ===== METHOD COMPARISON =====
    cat("METHOD COMPARISON:\n")
    cat(sprintf("  Difference in R:      %6.3f\n", multiple_correlation_results$R_difference))
    cat(sprintf("  Agreement level:      %s\n", multiple_correlation_results$method_agreement))
    
    if (multiple_correlation_results$method_agreement == "HIGH") {
        cat("  → Both methods show similar results (primarily linear relationships)\n")
    } else if (multiple_correlation_results$method_agreement == "MODERATE") {
        cat("  → Some difference between methods (mixed linear/non-linear patterns)\n")
    } else {
        cat("  → Large difference between methods (strong non-linear relationships)\n")
    }
    cat("\n")
    
    cat("\n")
    
    # Individual scenario contributions (if available)
    if ("pearson_contributions" %in% names(multiple_correlation_results)) {
        cat("INDIVIDUAL SCENARIO CONTRIBUTIONS (Pearson):\n")
        cat("(How each scenario contributes to predicting image count - linear)\n")
        for (i in 1:length(multiple_correlation_results$pearson_contributions)) {
            scenario <- names(multiple_correlation_results$pearson_contributions)[i]
            contrib <- multiple_correlation_results$pearson_contributions[i]
            p_val <- multiple_correlation_results$pearson_scenario_p_values[i]
            significance <- ifelse(p_val < 0.05, "*", "")
            
            cat(sprintf("  %-25s: β = %7.4f (p = %.4f) %s\n", 
                       scenario, contrib, p_val, significance))
        }
        cat("\n")
    }
    
    if ("spearman_contributions" %in% names(multiple_correlation_results)) {
        cat("INDIVIDUAL SCENARIO CONTRIBUTIONS (Spearman):\n")
        cat("(How each scenario contributes to predicting image count - monotonic)\n")
        for (i in 1:length(multiple_correlation_results$spearman_contributions)) {
            scenario <- names(multiple_correlation_results$spearman_contributions)[i]
            contrib <- multiple_correlation_results$spearman_contributions[i]
            p_val <- multiple_correlation_results$spearman_scenario_p_values[i]
            significance <- ifelse(p_val < 0.05, "*", "")
            
            cat(sprintf("  %-25s: β = %7.4f (p = %.4f) %s\n", 
                       scenario, contrib, p_val, significance))
        }
        cat("\n")
    }
    
    cat("INTERPRETATION:\n")
    cat(sprintf("  Pearson analysis: %s linear correlation (R = %.3f, %.1f%% variance explained)\n", 
               p_strength, multiple_correlation_results$pearson_R, 
               multiple_correlation_results$pearson_R_squared * 100))
    cat(sprintf("  Spearman analysis: %s monotonic correlation (R = %.3f, %.1f%% variance explained)\n", 
               s_strength, multiple_correlation_results$spearman_R, 
               multiple_correlation_results$spearman_R_squared * 100))
    
    # Determine which method shows stronger relationship
    stronger_method <- ifelse(multiple_correlation_results$pearson_R > multiple_correlation_results$spearman_R, 
                             "Pearson (linear)", "Spearman (monotonic)")
    cat(sprintf("  Stronger relationship detected by: %s\n", stronger_method))
    
    # Statistical significance interpretation
    both_significant <- multiple_correlation_results$pearson_p_value < 0.05 && multiple_correlation_results$spearman_p_value < 0.05
    pearson_only <- multiple_correlation_results$pearson_p_value < 0.05 && multiple_correlation_results$spearman_p_value >= 0.05
    spearman_only <- multiple_correlation_results$pearson_p_value >= 0.05 && multiple_correlation_results$spearman_p_value < 0.05
    neither_significant <- multiple_correlation_results$pearson_p_value >= 0.05 && multiple_correlation_results$spearman_p_value >= 0.05
    
    if (both_significant) {
        cat("  → Both linear and monotonic relationships are STATISTICALLY SIGNIFICANT\n")
    } else if (pearson_only) {
        cat("  → Only LINEAR relationship is statistically significant\n")
    } else if (spearman_only) {
        cat("  → Only MONOTONIC relationship is statistically significant\n")
    } else {
        cat("  → Neither relationship is statistically significant\n")
    }
    
    cat("\n")
    cat("(* indicates p < 0.05 - statistically significant)\n")
    
    # Create output text
    pearson_strength <- multiple_correlation_results$pearson_R
    if (pearson_strength < 0.3) {
        p_strength <- "WEAK"
    } else if (pearson_strength < 0.7) {
        p_strength <- "MODERATE" 
    } else {
        p_strength <- "STRONG"
    }
    
    spearman_strength <- multiple_correlation_results$spearman_R
    if (spearman_strength < 0.3) {
        s_strength <- "WEAK"
    } else if (spearman_strength < 0.7) {
        s_strength <- "MODERATE" 
    } else {
        s_strength <- "STRONG"
    }
    
    output_lines <- c(
        "HIERARCHICAL PERFORMANCE MULTIPLE CORRELATION ANALYSIS",
        paste(rep("=", 80), collapse = ""),
        paste("Analysis Date:", Sys.time()),
        paste("Input File:", basename(csv_path)),
        paste("Taxonomic Level:", toupper(taxonomic_level)),
        paste("Metric Analyzed:", toupper(metric_name)),
        paste("Scenarios Analyzed:", length(scenario_cols)),
        "",
        "DATASET SUMMARY:",
        paste("  Categories analyzed:", n_categories),
        paste("  Complete cases used:", multiple_correlation_results$n_complete_cases),
        paste("  Total images:", total_images),
        paste("  Image count statistics:"),
        paste("    Mean:", sprintf("%.2f", mean_count)),
        paste("    Median:", sprintf("%.0f", median_count)),
        paste("    Range:", paste(min_count, "-", max_count)),
        "",
        "MULTIPLE CORRELATION ANALYSIS:",
        paste("Multiple correlation between number of images and all", toupper(metric_name), "scenarios"),
        paste(rep("-", 80), collapse = ""),
        "",
        "PEARSON MULTIPLE CORRELATION (Linear Relationships):",
        paste("  Multiple R:", sprintf("%6.3f", multiple_correlation_results$pearson_R), 
              paste("(", p_strength, "correlation)")),
        paste("  R-squared:", sprintf("%6.3f", multiple_correlation_results$pearson_R_squared),
              paste("(", sprintf("%.1f%%", multiple_correlation_results$pearson_R_squared * 100), "of variance explained)")),
        ifelse(!is.na(multiple_correlation_results$pearson_adjusted_R_squared),
               paste("  Adjusted R-squared:", sprintf("%6.3f", multiple_correlation_results$pearson_adjusted_R_squared),
                     paste("(", sprintf("%.1f%%", multiple_correlation_results$pearson_adjusted_R_squared * 100), "adjusted for complexity)")),
               ""),
        ifelse(!is.na(multiple_correlation_results$pearson_f_statistic),
               paste("  F-statistic:", sprintf("%6.2f", multiple_correlation_results$pearson_f_statistic)),
               ""),
        paste("  p-value:", sprintf("%6.4f", multiple_correlation_results$pearson_p_value),
              ifelse(multiple_correlation_results$pearson_p_value < 0.05, "*", "")),
        "",
        "SPEARMAN MULTIPLE CORRELATION (Monotonic Relationships):",
        paste("  Multiple R:", sprintf("%6.3f", multiple_correlation_results$spearman_R), 
              paste("(", s_strength, "correlation)")),
        paste("  R-squared:", sprintf("%6.3f", multiple_correlation_results$spearman_R_squared),
              paste("(", sprintf("%.1f%%", multiple_correlation_results$spearman_R_squared * 100), "of variance explained)")),
        ifelse(!is.na(multiple_correlation_results$spearman_adjusted_R_squared),
               paste("  Adjusted R-squared:", sprintf("%6.3f", multiple_correlation_results$spearman_adjusted_R_squared),
                     paste("(", sprintf("%.1f%%", multiple_correlation_results$spearman_adjusted_R_squared * 100), "adjusted for complexity)")),
               ""),
        ifelse(!is.na(multiple_correlation_results$spearman_f_statistic),
               paste("  F-statistic:", sprintf("%6.2f", multiple_correlation_results$spearman_f_statistic)),
               ""),
        paste("  p-value:", sprintf("%6.4f", multiple_correlation_results$spearman_p_value),
              ifelse(multiple_correlation_results$spearman_p_value < 0.05, "*", "")),
        "",
        "METHOD COMPARISON:",
        paste("  Difference in R:", sprintf("%6.3f", multiple_correlation_results$R_difference)),
        paste("  Agreement level:", multiple_correlation_results$method_agreement),
        ""
    )
    
    # Add individual scenario contributions if available
    if ("pearson_contributions" %in% names(multiple_correlation_results)) {
        output_lines <- c(output_lines,
            "INDIVIDUAL SCENARIO CONTRIBUTIONS (Pearson):",
            "(How each scenario contributes to predicting image count - linear)",
            ""
        )
        
        for (i in 1:length(multiple_correlation_results$pearson_contributions)) {
            scenario <- names(multiple_correlation_results$pearson_contributions)[i]
            contrib <- multiple_correlation_results$pearson_contributions[i]
            p_val <- multiple_correlation_results$pearson_scenario_p_values[i]
            significance <- ifelse(p_val < 0.05, "*", "")
            
            output_lines <- c(output_lines,
                paste("  ", scenario, ": β =", sprintf("%7.4f", contrib), 
                      paste("(p =", sprintf("%.4f", p_val), ")", significance))
            )
        }
        output_lines <- c(output_lines, "")
    }
    
    if ("spearman_contributions" %in% names(multiple_correlation_results)) {
        output_lines <- c(output_lines,
            "INDIVIDUAL SCENARIO CONTRIBUTIONS (Spearman):",
            "(How each scenario contributes to predicting image count - monotonic)",
            ""
        )
        
        for (i in 1:length(multiple_correlation_results$spearman_contributions)) {
            scenario <- names(multiple_correlation_results$spearman_contributions)[i]
            contrib <- multiple_correlation_results$spearman_contributions[i]
            p_val <- multiple_correlation_results$spearman_scenario_p_values[i]
            significance <- ifelse(p_val < 0.05, "*", "")
            
            output_lines <- c(output_lines,
                paste("  ", scenario, ": β =", sprintf("%7.4f", contrib), 
                      paste("(p =", sprintf("%.4f", p_val), ")", significance))
            )
        }
        output_lines <- c(output_lines, "")
    }
    
    stronger_method <- ifelse(multiple_correlation_results$pearson_R > multiple_correlation_results$spearman_R, 
                             "Pearson (linear)", "Spearman (monotonic)")
    
    both_significant <- multiple_correlation_results$pearson_p_value < 0.05 && multiple_correlation_results$spearman_p_value < 0.05
    pearson_only <- multiple_correlation_results$pearson_p_value < 0.05 && multiple_correlation_results$spearman_p_value >= 0.05
    spearman_only <- multiple_correlation_results$pearson_p_value >= 0.05 && multiple_correlation_results$spearman_p_value < 0.05
    neither_significant <- multiple_correlation_results$pearson_p_value >= 0.05 && multiple_correlation_results$spearman_p_value >= 0.05
    
    output_lines <- c(output_lines,
        "INTERPRETATION:",
        paste("  Pearson analysis:", p_strength, "linear correlation (R =", sprintf("%.3f", multiple_correlation_results$pearson_R), 
              ",", sprintf("%.1f%%", multiple_correlation_results$pearson_R_squared * 100), "variance explained)"),
        paste("  Spearman analysis:", s_strength, "monotonic correlation (R =", sprintf("%.3f", multiple_correlation_results$spearman_R),
              ",", sprintf("%.1f%%", multiple_correlation_results$spearman_R_squared * 100), "variance explained)"),
        paste("  Stronger relationship detected by:", stronger_method),
        ifelse(both_significant,
               "  → Both linear and monotonic relationships are STATISTICALLY SIGNIFICANT",
               ifelse(pearson_only, "  → Only LINEAR relationship is statistically significant",
                     ifelse(spearman_only, "  → Only MONOTONIC relationship is statistically significant",
                           "  → Neither relationship is statistically significant"))),
        "",
        "STATISTICAL NOTES:",
        "  * indicates statistical significance (p < 0.05)",
        "  Pearson: measures linear relationships (traditional multiple correlation)",
        "  Spearman: measures monotonic relationships (robust to outliers and non-linearity)",
        "  Multiple R measures how well all scenarios together correlate with image count",
        "  R-squared shows the percentage of variance explained by the scenarios",
        "  Adjusted R-squared adjusts for the number of variables (prevents overfitting)",
        "  β coefficients show individual scenario contributions",
        "  Correlation strength: R < 0.3 (weak), 0.3-0.7 (moderate), > 0.7 (strong)",
        "  Agreement level: HIGH (diff < 0.1), MODERATE (0.1-0.2), LOW (> 0.2)",
        "",
        "DETAILED DATA (Complete Cases Only):",
        paste("Category,", "Count,", paste(scenario_cols, collapse = ","))
    )
    
    # Add detailed data for complete cases only
    for (i in 1:nrow(complete_data)) {
        row <- complete_data[i, ]
        scenario_values <- sapply(scenario_cols, function(s) sprintf("%.4f", row[[s]]))
        output_lines <- c(output_lines, 
                         paste(c(row[[taxonomic_level]], row$number_of_images, scenario_values), collapse = ","))
    }
    
    # Write to output file
    dir.create(dirname(output_txt), recursive = TRUE, showWarnings = FALSE)
    writeLines(output_lines, output_txt)
    
    cat("\nResults saved to:", output_txt, "\n")
    cat(paste(rep("=", 80), collapse = ""), "\n")
    
    # Return multiple correlation results
    return(multiple_correlation_results)
}

# Function to analyze a single metric CSV file (new format)
analyze_metric_csv <- function(csv_path, output_txt, metric_name) {
    # Analyze correlations for a single metric CSV file containing multiple scenarios.
    # 
    # Args:
    #   csv_path (str): Path to the metric CSV file (1 CSV = 1 metric, multiple scenarios as columns)
    #   output_txt (str): Path to save the correlation results  
    #   metric_name (str): Name of the metric for display purposes ('F1', 'precision', 'recall', 'mean_IoU')
    
    if (!file.exists(csv_path)) {
        cat("Warning: File not found:", csv_path, "\n")
        return(NULL)
    }
    
    # Extract taxonomic level from filename (assuming format: comparison_LEVEL_METRIC.csv)
    filename <- basename(csv_path)
    if (grepl("comparison_", filename)) {
        parts <- strsplit(filename, "_")[[1]]
        if (length(parts) >= 3) {
            taxonomic_level <- parts[2]
        } else {
            taxonomic_level <- "unknown"
        }
    } else {
        taxonomic_level <- "unknown"
    }
    
    cat("\n")
    cat(paste(rep("#", 80), collapse = ""), "\n")
    cat("ANALYZING METRIC:", toupper(metric_name), "- LEVEL:", toupper(taxonomic_level), "\n")
    cat(paste(rep("#", 80), collapse = ""), "\n")
    
    tryCatch({
        result <- compute_scenario_correlations(csv_path, output_txt, metric_name)
        return(result)
    }, error = function(e) {
        cat("Error analyzing", metric_name, ":", e$message, "\n")
        return(NULL)
    })
}

# Example usage
if (TRUE) {  # Set to TRUE to run examples
    # Single metric analysis (new format)
    csv_path <- "validation/hierarchical_metrics/comparisons/comparison_class_F1.csv"
    output_file <- "validation/hierarchical_metrics/correlations/correlation_class_F1.txt"
    result <- analyze_metric_csv(csv_path, output_file, "F1")
    
    # Analyze all metric files in the comparisons directory
    metrics <- c("F1", "precision", "recall", "mean_IoU")
    levels <- c("class", "order")
    
    for (level in levels) {
        for (metric in metrics) {
            csv_file <- paste0("validation/hierarchical_metrics/comparisons/comparison_", level, "_", metric, ".csv")
            output_file <- paste0("validation/hierarchical_metrics/correlations/correlation_", level, "_", metric, ".txt")
            
            if (file.exists(csv_file)) {
                analyze_metric_csv(csv_file, output_file, metric)
            } else {
                cat("File not found:", csv_file, "\n")
            }
        }
    }
}
