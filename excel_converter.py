import pandas as pd
import json
import numpy as np

class ExcelToJSONConverter:
    def __init__(self):
        pass

    def convert(self, df, unique_key, template_item=None):
        """
        Converts a DataFrame into a nested JSON structure.
        If 'template_item' is provided (a sample dict from master data), 
        it uses it to decide which columns belong to 'line_items' and which are headers.
        """
        if unique_key not in df.columns:
            raise ValueError(f"Unique Key '{unique_key}' not found in DataFrame columns.")

        # Replace NaNs with None for JSON compatibility
        df = df.replace({np.nan: None})
        
        # Analyze Template if provided
        template_line_keys = set()
        template_header_keys = set()
        if template_item:
            # Get top level keys
            template_header_keys = set(template_item.keys())
            # Check for line_items
            if 'line_items' in template_item and isinstance(template_item['line_items'], list):
                if len(template_item['line_items']) > 0:
                    template_line_keys = set(template_item['line_items'][0].keys())
        
        json_output = []
        grouped = df.groupby(unique_key)
        
        for key_val, group in grouped:
            obj = {}
            line_items = []
            
            rows = group.to_dict('records')
            
            # Determine Header vs Line Item fields
            header_cols = []
            line_cols = []
            
            # If template exists, use strict mapping
            if template_line_keys:
                for col in df.columns:
                    # If col is in template line keys (or similar?), put in line checking
                    # Simple case: check exact match or strict override
                    if col in template_line_keys and col not in template_header_keys:
                        line_cols.append(col)
                    elif col == 'line_items': 
                        pass # Should not happen in excel but just in case
                    else:
                        header_cols.append(col)
                        # But wait, what if a col is in BOTH? (rare, usually inconsistent)
            else:
                # Fallback to automatic variance detection
                # We use a "Global Variance" strategy:
                # If a column varies in ANY group (invoice) in the entire file, it is considered a Line Item Column.

                # Pre-calculate line columns (doing this once per convert call is efficient enough)
                if 'detected_line_cols' not in locals():
                    detected_line_cols = set()
                    for c in df.columns:
                        if c == unique_key: 
                            continue
                        # Check if max unique values in any group > 1
                        if df.groupby(unique_key)[c].nunique(dropna=False).max() > 1:
                            detected_line_cols.add(c)

                for col in df.columns:
                    if col in detected_line_cols:
                        line_cols.append(col)
                    else:
                        header_cols.append(col)
            
            # Build Header (from first row)
            first_row = rows[0]
            for col in header_cols:
                obj[col] = first_row[col]
            
            # Build Line Items
            # If template is used, we enforce line items structure if line_cols exist
            # Or if standard detection found varying cols
            
            # If template has line items, we ALWAYS iterate rows for line items, even if just 1 row
            target_line_cols = line_cols if line_cols else (list(template_line_keys) if template_line_keys else [])
            
            # Refined strategy:
            # If we have line_cols, we create line items.
            # If we have NO line_cols but template expects line items, what do we do?
            # We should probably put ALL non-header cols into line items?
            
            for row in rows:
                item = {}
                for col in line_cols: # Only populate detected line columns
                    item[col] = row[col]
                
                # If we have a template but line_cols was empty (maybe single row?), 
                # but template has line_items, we might want to force creating a line item?
                # Let's stick to "if we identified line columns, add them".
                
                if item: 
                    line_items.append(item)
            
            if line_items:
                obj['line_items'] = line_items
            
            json_output.append(obj)
            
        # Custom Type Adapter for JSON
        def json_serial(obj):
            """JSON serializer for objects not serializable by default json code"""
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if pd.isna(obj):
                return None
            return str(obj)

        return json.dumps(json_output, indent=2, default=json_serial)
