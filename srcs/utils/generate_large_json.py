import json
import random
import string
from pathlib import Path

def generate_large_json(file_path, num_entries=10000):
    """
    Generates a large JSON file with sample data.

    :param file_path: Path to the output JSON file.
    :param num_entries: Number of entries to generate in the JSON file.
    """
    data = []

    for i in range(num_entries):
        entry = {
            "id": i,
            "name": ''.join(random.choices(string.ascii_letters, k=10)),
            "email": f"user{i}@example.com",
            "age": random.randint(18, 80),
            "address": {
                "street": f"{random.randint(100, 999)} {random.choice(['Main St', 'Second Ave', 'Third Blvd'])}",
                "city": random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
                "zipcode": f"{random.randint(10000, 99999)}"
            },
            "phone": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
            "registered": random.choice([True, False]),
            "tags": [f"tag{random.randint(1, 10)}" for _ in range(random.randint(1, 5))]
        }
        data.append(entry)

    # Write the data to a JSON file
    with file_path.open('w') as file:
        json.dump(data, file, indent=2)

    print(f"Large JSON file generated at {file_path}")

def main():
    # Define the output file path
    output_file = Path("large_test_data.json")

    # Generate the large JSON file
    generate_large_json(output_file)

if __name__ == "__main__":
    main()
