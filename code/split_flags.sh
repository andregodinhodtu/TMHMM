#!/bin/bash

# 1. Check for correct arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_file> <flag1> [flag2] [flag3] ..."
    exit 1
fi

INPUT_FILE="$1"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found!"
    exit 1
fi

# 2. Isolate the flags
shift

# 3. Loop through flags and extract
for FLAG in "$@"; do
    echo "Extracting sequences for flag: $FLAG..."
    OUTPUT_FILE="${FLAG}_sequences.3line" 
    
    awk -v target_flag="$FLAG" '
    NR % 3 == 1 { 
        header = $0 
        # Strip hidden Windows carriage returns
        sub(/\r$/, "", header)
        
        # Split the header at the "|" and store the flag in a variable
        split(header, parts, "|")
        current_flag = parts[2]
        
        # Trim any accidental spaces from the extracted flag
        gsub(/^[ \t]+|[ \t]+$/, "", current_flag)
    }
    NR % 3 == 2 { seq = $0 }
    NR % 3 == 0 {
        third = $0
        
        # Exact string comparison! No regex special characters to worry about.
        if (current_flag == target_flag) {
            print header
            print seq
            print third
        }
    }' "$INPUT_FILE" > "$OUTPUT_FILE"
    
    echo "  -> Saved to $OUTPUT_FILE"
done

echo "All extractions complete!"
