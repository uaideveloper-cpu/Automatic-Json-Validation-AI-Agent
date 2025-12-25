import pandas as pd
import json
import re
import string

class ManualDataValidator:
    def __init__(self):
        pass

    def normalize_text(self, text):
        """
        Aggressively normalize text to ignore format differences.
        - Converts to lowercase
        - Removes all punctuation
        - Removes all whitespace
        """
        if text is None:
            return ""
        if isinstance(text, (list, dict)):
            return str(text).lower() # Simple stringification for complex types
        try:
            if pd.isna(text):
                return ""
        except ValueError:
            pass
        
        s = str(text).lower()
        
        # Helper to remove punctuation
        # We replace punctuation with empty string or space? 
        # "10,000" -> "10000" (good for numbers)
        # "2023-01-01" -> "20230101" (good for date comparison)
        # "Item - A" -> "Item A" or "ItemA"
        
        # Strategy: Remove punctuation entirely
        translator = str.maketrans('', '', string.punctuation)
        s = s.translate(translator)
        
        # Remove whitespace entirely to handle "10 kg" vs "10kg"
        s = re.sub(r'\s+', '', s)
        
        return s

    def validate(self, master_row, check_row):
        """
        Validates two rows manually (algorithmic comparison) and returns output 
        in the exact same format as the AI validator.
        """
        differences = {}
        match_status = "YES"
        confidence = "HIGH"
        explanation = "Manual validation successful. Records match."
        
        # Identify all unique keys from both rows
        all_keys = set(master_row.keys()) | set(check_row.keys())
        
        match_count = 0
        total_fields = 0

        for key in all_keys:
            # Skip internal keys if strict, but we compare everything present
            val_master = master_row.get(key)
            val_check = check_row.get(key)
            
            # Use recursive comparison helper
            is_match = self.compare_values(val_master, val_check)

            total_fields += 1
            if is_match:
                match_count += 1
            else:
                match_status = "NO"
                differences[key] = {
                    "master_value": str(val_master),
                    "check_value": str(val_check),
                    "analysis": "Value mismatch (normalized diff)"
                }

        # Determine final status and explanation
        if match_status == "NO":
            explanation = f"Manual validation found {len(differences)} mismatches."
            # If significant mismatches, confidence remains HIGH because it's deterministic.
        
        # JSON Output format mimicking AI
        result = {
            "match": match_status,
            "confidence": confidence,
            "explanation": explanation,
            "differences": differences,
            "summary": f"{match_count}/{total_fields} fields matched"
        }
        return result

    def compare_values(self, v1, v2):
        # 1. Recursive List Handling
        if isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2): return False
            for i in range(len(v1)):
                if not self.compare_values(v1[i], v2[i]):
                    return False
            return True
        
        # 2. Recursive Dict Handling with Fuzzy Key Matching
        if isinstance(v1, dict) and isinstance(v2, dict):
            # Helper to normalize keys (remove 'item_' prefix, lowercase, strip)
            def clean_key(k):
                k_str = str(k).lower()
                clean = k_str.replace('item_', '').replace('item', '').strip('_')
                return clean

            # Create maps of {clean_key: original_key}
            k1_map = {clean_key(k): k for k in v1.keys()}
            k2_map = {clean_key(k): k for k in v2.keys()}
            
            # RELAXED LOGIC: Only compare keys present in BOTH (Intersection)
            # This allows Master to have extra fields (like hsn_sac) without failing validation
            # and allows Check to have extra fields without failing (if that's desired, or we can restrict).
            # Plan specified: "Relax dictionary comparison to intersection of keys".
            common_clean_keys = set(k1_map.keys()) & set(k2_map.keys())
            
            if not common_clean_keys:
                # If no common keys, but both are not empty, might be a mismatch? 
                # If one is empty and other isn't, intersection is empty -> True (matches nothing effectively)
                # This seems safe for "ignore extra fields".
                pass

            for ck in common_clean_keys:
                orig_k1 = k1_map.get(ck)
                orig_k2 = k2_map.get(ck)
                
                val1 = v1.get(orig_k1)
                val2 = v2.get(orig_k2)
                
                if not self.compare_values(val1, val2):
                    return False
            return True

        # 3. Handle Nulls/NaNs safe check
        try:
            if pd.isna(v1) and pd.isna(v2): return True
            # Treat None and empty string as equal? "None" vs ""
            # normalizing later handles "" vs None usually if we define it so.
        except (ValueError, TypeError):
             if v1 is None and v2 is None: return True

        # 4. Date Comparison using Pandas (Moved UP priority and made robust)
        # Check if they look like dates before converting scalar numbers
        if not isinstance(v1, (int, float)) and not isinstance(v2, (int, float)):
             try:
                 # Debug prints to trace values in the console
                 print(f"DEBUG_CHECK: '{v1}' ({type(v1)}) vs '{v2}' ({type(v2)})")
                 dt1 = pd.to_datetime(v1, dayfirst=True, errors='coerce')
                 dt2 = pd.to_datetime(v2, dayfirst=True, errors='coerce')
                 
                 if pd.notna(dt1) or pd.notna(dt2):
                     print(f"DEBUG_PARSED: dt1={dt1}, dt2={dt2}")

                 # Check if both are valid dates
                 if pd.notna(dt1) and pd.notna(dt2):
                     # Compare just the date part to ignore time differences
                     if dt1.date() == dt2.date():
                         return True
                     # If years match but format differed (e.g. 25 vs 2025), pandas handles that.
             except Exception as e:
                 print(f"DEBUG_ERROR: {e}")
                 pass

        # 5. Scalar Float Comparison
        try:
            s1 = str(v1).replace(',', '')
            s2 = str(v2).replace(',', '')
            f1 = float(s1)
            f2 = float(s2)
            # Use a small epsilon
            if abs(f1 - f2) < 0.01:
                return True
        except (ValueError, TypeError):
            pass

        # 6. Fallback: Aggressive String Normalization
        return self.normalize_text(v1) == self.normalize_text(v2)
