#!/usr/bin/env python3
"""
Error Analysis Script for Text2SQL Evaluation Results

Analyzes CSV files containing text2sql evaluation results and adds error analysis
for rows where execution_accuracy is incorrect using OpenAI's GPT model.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import dotenv
import pandas as pd
from openai import OpenAI

dotenv.load_dotenv("../../../.env")

ERROR_TAXONOMY = [
    "AGGR_DISTINCT_MISSING",
    "WRONG_FILTER_COLUMN", 
    "WRONG_SOURCE_TABLE_OR_COLUMN",
    "EXTRA_TRANSFORMATION_OR_CONDITION",
    "OUTPUT_COLUMN_ALIAS_MISMATCH",
    "NULL_OR_EMPTY_RESULT",
    "GENERIC_VALUE_MISMATCH",
    "OTHER"
]


def get_error_analysis(client: OpenAI, row: Dict[str, Any]) -> Dict[str, Any]:
    """Get error analysis from OpenAI for a single row."""
    
    prompt = f"""You are analyzing why a Text2SQL prediction failed. Given the following information, identify the error codes and provide a brief analysis.

Available error codes:
- AGGR_DISTINCT_MISSING: Used COUNT/SUM without DISTINCT or deduplication
- WRONG_FILTER_COLUMN: Filtered on the wrong column
- WRONG_SOURCE_TABLE_OR_COLUMN: Selected metric from the wrong table/column
- EXTRA_TRANSFORMATION_OR_CONDITION: Added ABS(), extra filters that change results
- OUTPUT_COLUMN_ALIAS_MISMATCH: Output column names don't match
- NULL_OR_EMPTY_RESULT: Result is None/empty due to wrong filters or source
- GENERIC_VALUE_MISMATCH: Aggregation computed but numeric value differs for unclear reasons
- OTHER: Fallback

Query: {row['query']}
Expected SQL: {row['expected_sql']}
Predicted SQL: {row['predicted_sql']}
SQL Validity: {row['sql_validity']}
Execution Accuracy: {row['execution_accuracy']}
Validity Reason: {row['validity_reason']}
Accuracy Reason: {row['accuracy_reason']}

Respond with JSON containing:
- error_codes: array of applicable error codes (1 or more)
- error_analysis: brief 1-3 sentence explanation of what went wrong"""

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    
    content = response.choices[0].message.content
    if content is None:
        return {"error_codes": ["OTHER"], "error_analysis": "No response from model"}
    
    return json.loads(content)




def analyze_errors(input_file: str, output_file: str) -> None:
    """Analyze errors in the CSV file and add error analysis columns."""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    client = OpenAI()
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Initialize new columns
    df['error_analysis'] = ''
    df['error_codes'] = ''
    
    # Process rows with incorrect execution accuracy
    incorrect_mask = df['execution_accuracy'].str.lower() == 'incorrect'
    incorrect_rows = df[incorrect_mask]
    
    print(f"Found {len(incorrect_rows)} rows with incorrect execution accuracy")
    
    # Process rows sequentially
    total_rows = len(incorrect_rows)
    for i, (idx, row) in enumerate(incorrect_rows.iterrows(), 1):
        print(f"Processing row {i}/{total_rows} (ID: {row.get('id', 'unknown')})")
        
        try:
            result = get_error_analysis(client, row.to_dict())
            df.at[idx, 'error_analysis'] = result.get('error_analysis', 'Analysis not available')
            df.at[idx, 'error_codes'] = json.dumps(result.get('error_codes', ['OTHER']))
            print(f"  ✓ Completed: {result.get('error_codes', ['OTHER'])}")
        except Exception as e:
            print(f"  ✗ Error processing row {idx}: {e}")
            df.at[idx, 'error_analysis'] = f"Error during analysis: {str(e)}"
            df.at[idx, 'error_codes'] = json.dumps(["OTHER"])
    
    # Write the output CSV
    df.to_csv(output_file, index=False)
    print(f"Analysis complete. Output written to: {output_file}")
    
    # Print error code summary
    print("\n" + "="*50)
    print("ERROR CODE SUMMARY")
    print("="*50)
    
    error_counts = {}
    for _, row in df[incorrect_mask].iterrows():
        try:
            error_codes_str = str(row['error_codes']).strip()
            if error_codes_str and error_codes_str != 'nan':
                codes = json.loads(error_codes_str)
                for code in codes:
                    error_counts[code] = error_counts.get(code, 0) + 1
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            error_counts['OTHER'] = error_counts.get('OTHER', 0) + 1
    
    if error_counts:
        for code, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{code:<35} {count:>3}")
    else:
        print("No error codes found.")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Analyze errors in Text2SQL evaluation results")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", help="Output CSV file path (default: <input>_annotated.csv)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    if args.output:
        output_path = args.output
    else:
        output_path = input_path.parent / f"{input_path.stem}_annotated.csv"
    
    analyze_errors(args.input, str(output_path))


if __name__ == "__main__":
    main()
