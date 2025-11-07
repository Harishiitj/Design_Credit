#!/usr/bin/env python3
"""
Parse label variable files and create a features JSON file.

This script processes .txt files containing Stata label variable definitions
and extracts them into a structured JSON format.

Usage:
    python parse_label_variables.py

Configuration:
    Edit the input_files list in the __main__ section to specify which files to parse.
"""

import re
import json
from pathlib import Path
from collections import defaultdict


def parse_label_variable_files(file_list):
    """
    Parse multiple .txt files and extract label variable information.
    
    The function looks for lines matching the pattern:
        label variable <key_name> "<description>"
    
    Args:
        file_list: List of filenames to parse (e.g., ["IAIR7EFL.txt", "IAMR7EFL.txt"])
        
    Returns:
        List of dictionaries with keys: key_name, description, file
    """
    # Dictionary to store features: key_name -> {description, files}
    features_dict = defaultdict(lambda: {"description": "", "files": set()})
    
    # Regular expression to match lines like: label variable v006     "Month of interview"
    # Pattern explanation:
    #   ^label\s+variable\s+ : matches "label variable" at start of line with whitespace
    #   (\S+)                : captures the key_name (non-whitespace characters)
    #   \s+                  : matches whitespace
    #   "([^"]+)"            : captures the description inside quotes
    pattern = r'^label\s+variable\s+(\S+)\s+"([^"]+)"'
    
    for filename in file_list:
        try:
            # Extract the .DTA filename from the .txt filename
            # e.g., "IAIR7EFL.txt" -> "IAIR7EFL.DTA"
            dta_filename = Path(filename).stem + ".DTA"
            
            print(f"Processing {filename}...")
            
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = 0
                match_count = 0
                
                for line in f:
                    line_count += 1
                    match = re.match(pattern, line)
                    
                    if match:
                        key_name = match.group(1)
                        description = match.group(2)
                        
                        # Add or update the feature
                        # If the same feature appears in multiple files, it will be added to the file list
                        features_dict[key_name]["description"] = description
                        features_dict[key_name]["files"].add(dta_filename)
                        match_count += 1
                
                print(f"  - Processed {line_count} lines, found {match_count} label variables")
        
        except FileNotFoundError:
            print(f"Warning: File {filename} not found, skipping...")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Convert to final format
    # Sort by key_name for consistent output
    features_list = []
    for key_name in sorted(features_dict.keys()):
        features_list.append({
            "key_name": key_name,
            "description": features_dict[key_name]["description"],
            "file": sorted(list(features_dict[key_name]["files"]))
        })
    
    return features_list


def save_to_json(features, output_filename="features.json"):
    """
    Save features to JSON file with pretty formatting.
    
    Args:
        features: List of feature dictionaries
        output_filename: Name of the output JSON file
    """
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully saved {len(features)} features to {output_filename}")


def main():
    """Main execution function."""
    
    # ============================================================
    # CONFIGURATION: Add your input files here
    # ============================================================
    
    input_files = [
        "IABR7EFL.txt",
        "IACR7EFL.txt",
        "IAHR7EFL.txt",
        "IAIR7EFL.txt",
        "IAKR7EFL.txt",
        "IAMR7EFL.txt",
    ]
    
    # ============================================================
    # PROCESSING
    # ============================================================
    print("="*60)
    print("Label Variable Parser")
    print("="*60)
    print(f"\nFiles to process: {len(input_files)}")
    for f in input_files:
        print(f"  - {f}")
    print()
    
    # Parse all files
    features = parse_label_variable_files(input_files)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Found {len(features)} unique features across all files")
    print(f"{'='*60}\n")
    
    # Save to JSON
    save_to_json(features, "features.json")
    
    # Display sample entries
    print("\nSample output (first 5 entries):")
    print("-" * 60)
    for feature in features[:5]:
        print(json.dumps(feature, indent=2))
        print()


if __name__ == "__main__":
    main()
