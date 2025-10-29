import pandas as pd
import os
import sys

def generate_latex_table(csv_path, output_path=None):
    """
    Generate LaTeX table code from a comparison CSV file.
    
    Args:
        csv_path: Path to the comparison CSV file
        output_path: Optional path to save the LaTeX code (if None, prints to stdout)
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract metadata from filename
    filename = os.path.basename(csv_path)
    # Expected format: comparison_<level>_<metric>.csv
    parts = filename.replace('.csv', '').split('_')
    level = parts[1]  # 'class' or 'order'
    metric = parts[2]  # 'F1' or 'mean_IoU' or 'meanIoU'
    
    # Format metric name for display
    metric_display = 'F1 scores' if metric == 'F1' else 'mean IoU scores'
    metric_short = 'F1' if metric == 'F1' else 'mean IoU'
    
    # Get the taxonomic level column name and number of images column
    level_col = level
    num_images_col = 'number_of_images'
    
    # Get scenario columns (exclude metadata columns)
    exclude_cols = [level_col, num_images_col, 'mean', 'std', 'min', 'max', 'range', 'best_scenario', 'best_value']
    scenario_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Replace underscores with spaces and remove metric suffix from scenario names
    scenario_display_names = {}
    for col in scenario_cols:
        # Remove the metric suffix (e.g., '_F1', '_mean_IoU', '_meanIoU')
        display_name = col
        # Try different variations of metric suffixes
        for suffix in [f'_{metric}', '_mean_IoU', '_meanIoU', '_mean IoU']:
            if display_name.endswith(suffix):
                display_name = display_name[:-len(suffix)]
                break
        # Replace remaining underscores with spaces
        display_name = display_name.replace('_', ' ')
        scenario_display_names[col] = display_name
    
    # Prepare caption suffix based on truncation
    # (This will be updated after checking row count)
    caption_suffix = ''
    
    # Start building LaTeX code
    latex_lines = []
    latex_lines.append(r'\begin{table}[H]')
    latex_lines.append(r'\centering')
    # Placeholder for caption - will be updated after truncation check
    caption_placeholder_idx = len(latex_lines)
    latex_lines.append('')  # Placeholder
    latex_lines.append(f'\\label{{tab:comparison_{level}_{metric}}}')
    latex_lines.append(r'\footnotesize')
    
    # Build column specification
    num_cols = 2 + len(scenario_cols)  # level + num_images + scenarios
    col_spec = 'lr' + 'c' * len(scenario_cols)
    latex_lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    latex_lines.append(r'\toprule')
    
    # Build header row
    header_parts = [f'\\thead{{\\textbf{{{level.capitalize()}}}}}', 
                    r'\thead{\textbf{\# images}}']
    
    for scenario in scenario_cols:
        # Use display name with spaces instead of underscores
        display_name = scenario_display_names[scenario]
        
        # Split scenario name at spaces or '+' for multiline headers
        if '+' in display_name:
            parts = display_name.split('+')
            header = f'\\thead{{\\textbf{{{parts[0]}+}}\\\\\\textbf{{{parts[1]}}}}}'
        elif ' ' in display_name:
            parts = display_name.split(' ', 1)
            header = f'\\thead{{\\textbf{{{parts[0]}}}\\\\\\textbf{{{parts[1]}}}}}'
        else:
            header = f'\\thead{{\\textbf{{{display_name}}}}}'
        header_parts.append(header)
    
    latex_lines.append(' & '.join(header_parts) + ' \\\\')
    latex_lines.append(r'\midrule')
    
    # Sort by number_of_images descending (should already be sorted, but ensure it)
    df = df.sort_values(num_images_col, ascending=False).reset_index(drop=True)
    
    # Calculate averages across ALL rows before truncation
    averages = {}
    for col in scenario_cols:
        averages[col] = df[col].mean()
    
    # Limit to top 20 if there are more than 20 rows
    total_rows = len(df)
    if total_rows > 20:
        df = df.head(20)
        caption_suffix = f' (top 20 with most images out of {total_rows})'
    else:
        caption_suffix = ''
    
    # Update the caption with the suffix
    caption_text = f'\\caption{{{level.capitalize()}-level {metric_display} on the ArthroNat test set across different training scenarios{caption_suffix}}}'
    latex_lines[caption_placeholder_idx] = caption_text
    
    # Add data rows with alternating row colors
    for idx, row in df.iterrows():
        row_parts = []
        
        # Add taxonomic name
        row_parts.append(row[level_col])
        
        # Add number of images
        row_parts.append(str(int(row[num_images_col])))
        
        # Find the best value for this row to bold it
        scenario_values = [row[col] for col in scenario_cols]
        max_value = max(scenario_values)
        
        # Add scenario values, bolding the best ones
        for col in scenario_cols:
            value = row[col]
            formatted_value = f'{value:.2f}'
            
            # Bold if it's the maximum (or within 0.001 of max for floating point comparison)
            if abs(value - max_value) < 0.001:
                formatted_value = f'\\textbf{{{formatted_value}}}'
            
            row_parts.append(formatted_value)
        
        # Build the row with optional row color
        row_content = ' & '.join(row_parts) + ' \\\\'
        if idx % 2 == 0:
            row_content = r'\rowcolor{gray!10} ' + row_content
        
        latex_lines.append(row_content)
    
    # Add separator line before average row
    latex_lines.append(r'\midrule')
    
    # Add average row
    avg_row_parts = []
    # Handle plural form correctly (class -> classes, order -> orders)
    level_plural = 'classes' if level == 'class' else f'{level}s'
    avg_row_parts.append(f'Average on all {total_rows} {level_plural}')  # Taxonomic level text with count
    avg_row_parts.append('')  # Empty for number of images column
    
    # Find the best average value
    avg_values = [averages[col] for col in scenario_cols]
    max_avg = max(avg_values)
    
    # Add average values for each scenario, bolding the best one(s)
    for col in scenario_cols:
        avg_value = averages[col]
        formatted_avg = f'{avg_value:.2f}'
        
        # Bold if it's the maximum (or within 0.001 of max for floating point comparison)
        if abs(avg_value - max_avg) < 0.001:
            formatted_avg = f'\\textbf{{{formatted_avg}}}'
        
        avg_row_parts.append(formatted_avg)
    
    avg_row_content = ' & '.join(avg_row_parts) + ' \\\\'
    latex_lines.append(avg_row_content)
    
    # Close table
    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')
    
    # Join all lines
    latex_code = '\n'.join(latex_lines)
    
    # Output
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_code)
        print(f"LaTeX table saved to: {output_path}")
    else:
        print(latex_code)
    
    return latex_code

def generate_all_tables(comparison_dir='validation/hierarchical_metrics/comparisons', 
                        output_dir=None):
    """
    Generate LaTeX tables for all comparison CSV files in a directory.
    
    Args:
        comparison_dir: Directory containing comparison CSV files
        output_dir: Directory to save LaTeX table files (if None, prints to stdout)
    """
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("GENERATING LATEX TABLES")
    print(f"{'='*70}\n")
    
    # Find all comparison CSV files
    csv_files = [f for f in os.listdir(comparison_dir) if f.startswith('comparison_') and f.endswith('.csv')]
    
    if not csv_files:
        print(f"No comparison CSV files found in {comparison_dir}")
        return
    
    for csv_file in csv_files:
        csv_path = os.path.join(comparison_dir, csv_file)
        
        if output_dir:
            output_file = csv_file.replace('.csv', '.tex')
            output_path = os.path.join(output_dir, output_file)
        else:
            output_path = None
        
        print(f"Processing {csv_file}...")
        generate_latex_table(csv_path, output_path)
        print()
    
    if output_dir:
        print(f"{'='*70}")
        print(f"Generated {len(csv_files)} LaTeX tables in {output_dir}")
        print(f"{'='*70}\n")
    else:
        print(f"{'='*70}")
        print(f"Generated {len(csv_files)} LaTeX tables")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    # Check if a specific CSV file is provided as argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        generate_latex_table(csv_path, output_path)
    else:
        # Generate all tables
        generate_all_tables()
