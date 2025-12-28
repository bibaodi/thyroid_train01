#!/bin/bash

# Define a Bash variable
my_var="some_value"

# Pass the Bash variable to awk using the -v option
echo "Some text to process" | awk -v awk_var="$my_var" '{
  # Use the awk_var in your awk script
  print "Bash variable value: " awk_var
  print "Processing line: " $0
}'

