import pandas as pd
from collections import Counter

# Read the CSV file
file_path = 'dataset/all_verify_sop_with_predictions.csv'
df = pd.read_csv(file_path)

# Count the occurrences of each unique value in the 'type' column
type_counts = df['type'].value_counts()

# Display the results
print("Count of each unique value in the 'type' column:")
print(type_counts)

# Save the results to a text file
with open('type_column_statistics.txt', 'w') as f:
    f.write("Count of each unique value in the 'type' column:\n")
    f.write(str(type_counts))
    f.write("\n\nTotal unique values: " + str(len(type_counts)))
    f.write("\nTotal entries: " + str(len(df)))

print("\nResults have been saved to 'type_column_statistics.txt'")