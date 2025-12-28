import pandas as pd
import csv

def analyze_type_column(csv_file_path):
    """
    Analyze the 'type' column in a CSV file and generate statistics.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get value counts for the 'type' column
    type_counts = df['type'].value_counts()
    
    # Calculate percentages
    type_percentages = df['type'].value_counts(normalize=True) * 100
    
    # Combine counts and percentages
    stats_df = pd.DataFrame({
        'Count': type_counts,
        'Percentage': type_percentages
    })
    
    # Display results
    print("=" * 50)
    print("STATISTICAL ANALYSIS OF 'type' COLUMN")
    print("=" * 50)
    print(f"Total records: {len(df)}")
    print(f"Unique types: {len(type_counts)}")
    print("-" * 50)
    print(stats_df)
    print("-" * 50)
    
    # Save to CSV file
    stats_df.to_csv('type_column_detailed_stats.csv')
    print("Detailed statistics saved to 'type_column_detailed_stats.csv'")
    
    # Save summary to text file
    with open('type_column_summary.txt', 'w') as f:
        f.write("STATISTICAL ANALYSIS OF 'type' COLUMN\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Unique types: {len(type_counts)}\n")
        f.write("-" * 50 + "\n")
        f.write(stats_df.to_string())
        f.write("\n" + "-" * 50 + "\n")
    
    print("Summary saved to 'type_column_summary.txt'")

if __name__ == "__main__":
    csv_file_path = 'dataset/all_verify_sop_with_predictions.csv'
    analyze_type_column(csv_file_path)