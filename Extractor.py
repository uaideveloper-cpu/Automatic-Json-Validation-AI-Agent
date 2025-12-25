import pdfplumber
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class InvoiceDataExtractor:
    def __init__(self):
        # Common keywords for segmentation
        self.BLOCK_KEYWORDS = {
            "header_end": ["Bill to", "Ship to", "Buyer", "Consignee", "Tax Invoice"],
            "items_start": ["Sl No", "Description", "Particulars", "HSN"],
            "items_end": ["Total", "Amount Chargeable", "Tax Amount", "Grand Total", "Amount (in words)"],
        }

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Entry point for PDF extraction"""
        text = self._extract_text(pdf_path)
        return self.extract_from_text(text)

    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Main extraction logic using context-aware strategy"""
        if not text:
            return {}
        
        # 1. Normalize
        clean_text = self._normalize_text(text)
        
        # 2. Segment
        segments = self._segment_document(clean_text)
        
        # 3. Extract Block Data
        data = {
            "header_metadata": self._extract_header_metadata(segments["header"]),
            "invoice_metadata": self._extract_invoice_metadata(segments["header"]),
            "seller_address": self._extract_party_block(segments["seller"], "Seller"),
            "buyer_bill_to": self._extract_party_block(segments["buyer"], "Buyer"),
            "consignee_ship_to": self._extract_party_block(segments["consignee"], "Consignee"),
            "line_items": self._extract_line_items(segments["items"]),
            "tax_summary": self._extract_tax_summary(segments["taxes_totals"]),
            "totals": self._extract_totals(segments["taxes_totals"]),
            "raw_text": clean_text # Optional: keep for debug
        }
        
        # Fallback: if totals missing, try to infer from items
        if not data["totals"].get("grand_total") and data["line_items"]:
             total_amt = sum(item.get("amount", 0) for item in data["line_items"])
             if total_amt > 0:
                 data["totals"]["grand_total"] = total_amt # Approx
        
        return data

    def _extract_text(self, pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract text with layout preservation
                return "\n".join([p.extract_text(x_tolerance=1, y_tolerance=3) or "" for p in pdf.pages])
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def _normalize_text(self, text: str) -> str:
        """Clean and normalize text for consistent processing"""
        lines = text.splitlines()
        normalized_lines = []
        for line in lines:
            # Fix spacing around colons
            line = re.sub(r'\s*:\s*', ' : ', line)
            # Remove repeated spaces
            line = re.sub(r'\s+', ' ', line).strip()
            if line:
                normalized_lines.append(line)
        return "\n".join(normalized_lines)

    def _segment_document(self, text: str) -> Dict[str, str]:
        """Segment document into logical blocks based on keywords"""
        segments = {
            "header": "",
            "seller": "",
            "buyer": "",
            "consignee": "",
            "items": "",
            "taxes_totals": ""
        }
        
        lines = text.splitlines()
        
        # State machine for segmentation
        current_block = "header"
        
        # Markers
        buyer_markers = ["Buyer", "Bill To", "Billed To"]
        consignee_markers = ["Consignee", "Ship To", "Shipped To"]
        item_table_markers = ["Description of Goods", "Sl No", "HSN/SAC", "Particulars"]
        tax_total_markers = ["Total", "Amount Chargeable", "Tax Amount", "Grand Total", "Amount in Words"]
        
        # Heuristics:
        # Header is top until we see Party info.
        # Seller is usually implicit at top or explicitly marked. 
        # For this logic, we'll keep simple state transitions.
        
        buffer = []
        
        for i, line in enumerate(lines):
            # Check transitions
            
            # Transition to Buyer/Consignee
            is_buyer = any(m.lower() in line.lower() for m in buyer_markers)
            is_consignee = any(m.lower() in line.lower() for m in consignee_markers)
            
            # Items
            is_items = any(m.replace(" ", "").lower() in line.replace(" ", "").lower() for m in item_table_markers)
            
            # Totals (Weak check, needs context)
            is_totals = any(line.strip().startswith(m) for m in tax_total_markers) and (i > len(lines) * 0.5)

            if current_block == "header":
                if is_buyer: # Found buyer start
                    segments["header"] = "\n".join(buffer)
                    buffer = [line]
                    current_block = "buyer"
                    continue
                elif is_consignee:
                    segments["header"] = "\n".join(buffer)
                    buffer = [line]
                    current_block = "consignee"
                    continue
                elif "Tax Invoice" in line and len(buffer) > 5: # Maybe seller info is before "Tax Invoice"?
                    # Common in Indian invoices: Seller Name -> Tax Invoice -> Details
                    # Let's assume header contains everything until first Party marker
                    pass

            elif current_block in ["buyer", "consignee", "seller"]:
                # If we are in a party block and see another party marker
                if is_buyer and current_block != "buyer":
                    segments[current_block] = "\n".join(buffer)
                    buffer = [line]
                    current_block = "buyer"
                    continue
                elif is_consignee and current_block != "consignee":
                    segments[current_block] = "\n".join(buffer)
                    buffer = [line]
                    current_block = "consignee"
                    continue
                
                # Check for items start
                if is_items:
                    segments[current_block] = "\n".join(buffer)
                    buffer = [line]
                    current_block = "items"
                    continue
            
            elif current_block == "items":
                # Check for totals start
                # Usually purely based on keywords like "Total" or tax tables
                if is_totals or "Amount Chargeable" in line:
                    segments[current_block] = "\n".join(buffer)
                    buffer = [line]
                    current_block = "taxes_totals"
                    continue
            
            buffer.append(line)
        
        # Flush last buffer
        segments[current_block] = "\n".join(buffer)
        
        # Heuristic: If Seller empty, try to extract from Header (top lines)
        if not segments["seller"] and segments["header"]:
            # Crude split: First few lines of header are likely Seller
            header_lines = segments["header"].splitlines()
            # If explicit "Sold By" isn't found, assume top 3-4 lines are seller if they aren't Meta
            segments["seller"] = "\n".join(header_lines[:5]) # Take top 5 lines as candidate
        
        return segments

    def _extract_value(self, text: str, labels: List[str], regex_pattern: str = r".+") -> Optional[str]:
        """Generic label-based extractor"""
        for label in labels:
            # Pattern: Label +/- separators +/- Value
            # e.g., "Invoice No : 123" or "Invoice No - 123"
            escaped_label = re.escape(label)
            # Look for Label followed by optional separator and capture group
            pattern = rf"{escaped_label}\s*[:\-\.]?\s*({regex_pattern})"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                # filter out if val is too matched with symbols
                if re.match(r"^[:\-\.]+$", val): continue
                return val
        return None

    def _extract_header_metadata(self, text: str) -> Dict[str, Any]:
        return {
            "irn": self._extract_value(text, ["IRN", "IRN No"], r"[0-9a-fA-F]+"),
            "ack_no": self._extract_value(text, ["Ack No", "Acknowledgement No"], r"\d+"),
            "ack_date": self._extract_value(text, ["Ack Date", "Date"], r"[\d\w\-]+")
        }

    def _extract_invoice_metadata(self, text: str) -> Dict[str, Any]:
        return {
            "invoice_no": self._extract_value(text, ["Invoice No", "Inv No", "Bill No"], r"[^\s]+"),
            "invoice_date": self._extract_value(text, ["Invoice Date", "Dated", "Date"], r"\d{2}-[A-Za-z]{3}-\d{2}|\d{2}/\d{2}/\d{4}"),
            "disp_doc_no": self._extract_value(text, ["Dispatch Doc No", "Disp Doc no"], r"[^\s]+"),
            "dispatched_through": self._extract_value(text, ["Dispatched through", "Disp through"], r"[A-Za-z\s]+"),
            "destination": self._extract_value(text, ["Destination"], r"[A-Za-z\s]+"),
            "vehicle_no": self._extract_value(text, ["Motor Vehicle No", "Vehicle No"], r"[A-Z0-9]+"),
            "terms_payment": self._extract_value(text, ["Terms of Payment"], r"[^\n]+"),
        }

    def _extract_party_block(self, text: str, role: str) -> Dict[str, Any]:
        """Extract details from a party block (Seller/Buyer/Consignee)"""
        details = {
            "company_name": None, "gstin": None, "address": None,
            "state_name": None, "code": None, "pincode": None
        }
        
        if not text: return details
        
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        
        # GSTIN Pattern
        gstin_match = re.search(r"GSTIN/UIN\s*:\s*([A-Z0-9]+)", text, re.IGNORECASE)
        if gstin_match: details["gstin"] = gstin_match.group(1)
        
        # State
        state_match = re.search(r"State Name\s*:\s*([A-Za-z\s]+)", text, re.IGNORECASE)
        if state_match: details["state_name"] = state_match.group(1).replace(",", "").strip()
        
        code_match = re.search(r"Code\s*:\s*(\d+)", text, re.IGNORECASE)
        if code_match: details["code"] = code_match.group(1)

        # Company Name strategy:
        # First line that is NOT a label (like "Buyer:") and NOT a known meta field
        for line in lines:
            if role.upper() in line.upper() and len(line) < 20: continue # Skip "Buyer (Bill to)" header
            if "GSTIN" in line or "State Name" in line: continue
            
            # Heuristic: Valid company name usually has no numbers at start
            if re.match(r"^[A-Za-z]", line):
                details["company_name"] = line
                break
        
        # Address extraction:
        # Everything that isn't metadata or name
        addr_lines = []
        for line in lines:
            if line == details.get("company_name"): continue
            if any(k in line for k in ["GSTIN", "State Name", "Code :", "Buyer", "Consignee", "Bill to", "Ship to"]): continue
            addr_lines.append(line)
            
            # Pincode
            pin_match = re.search(r"\b(\d{6})\b", line)
            if pin_match: details["pincode"] = pin_match.group(1)

        details["address"] = ", ".join(addr_lines)
        return details

    def _extract_line_items(self, text: str) -> List[Dict[str, Any]]:
        items = []
        lines = text.splitlines()
        
        # Dynamic Header Detection
        # We look for a line containing "Description" and "Amount" or "Rate"
        header_idx = -1
        for i, line in enumerate(lines):
            if "Description" in line and ("Amount" in line or "Rate" in line):
                header_idx = i
                break
        
        if header_idx == -1: 
             # Fallback: assume start if no header found? Or empty.
             # If "items" block text exists, maybe it's just rows.
             if len(lines) > 0: header_idx = 0 # Try parsing all lines
        
        # Regex for row parsing
        # Expect: Sl No (opt) | Description | HSN (opt) | Qty | Unit | Rate | Amount
        # Broad pattern: Number ... Number ... Number
        
        current_item = None
        
        for i in range(header_idx + 1, len(lines)):
            line = lines[i].strip()
            if not line: continue
            
            # Stop if we hit totals line (double check)
            if "Total" in line and "words" not in line.lower() and re.search(r"[\d,]+\.\d{2}", line):
                 break

            # Try to match a standard row pattern
            # 1. Sl No (digits) at start OR Description text
            # 2. Look for numbers at the end (Amount, Rate, Qty)
            
            # Find all numbers at end of line (money/qty format)
            # Pattern: 1,234.00 or 1234.00
            numbers = re.findall(r"[\d,]+\.\d{2}", line)
            
            if len(numbers) >= 3: # Likely Amount, Rate, Qty are present
                # Assume structure: ... Qty ... Rate ... Amount
                # Pop from end
                try:
                    amount = float(numbers[-1].replace(",", ""))
                    rate = float(numbers[-2].replace(",", ""))
                    # Qty might be with unit "300 Ltrs"
                    
                    # Reconstruction for Description
                    # Remove the matched numbers and try to find description
                    # This is tricky with regex replacement, so let's split logic
                    
                    # Let's use a robust regex for the whole line if possible
                    # Pattern: (Sl)? (Desc) (HSN)? (Qty) (Unit)? (Rate) (Amount)
                    
                    # Find HSN (8 digits usually)
                    hsn_match = re.search(r"\b(\d{4,8})\b", line)
                    hsn = hsn_match.group(1) if hsn_match else ""
                    
                    # Extract Quantity
                    # Look for number before Rate
                    # Simple approach: use the numbers list
                    # If we have [qty, rate, amount]
                    qty_val = float(numbers[-3].replace(",", ""))
                    
                    # Description: Text before HSN or before Qty
                    # Split line by HSN or first number
                    parts = []
                    if hsn:
                        parts = line.split(hsn)
                    else:
                        # Split by first number found (Qty)
                        parts = line.split(numbers[-3])
                    
                    description = parts[0].strip()
                    # Clean Sl No from description (digits at start)
                    description = re.sub(r"^\d+\s+", "", description).strip()
                    
                    current_item = {
                        "sl_no": len(items) + 1,
                        "description": description,
                        "hsn_sac": hsn,
                        "quantity": qty_val,
                        "rate": rate,
                        "amount": amount,
                        "unit": "NOS" # default
                    }
                    
                    # Check unit in text (e.g., "Ltrs", "Kgs")
                    unit_match = re.search(r"\b(Ltrs|Kgs|Nos|Pcs|Box)\b", line, re.IGNORECASE)
                    if unit_match: current_item["unit"] = unit_match.group(1)
                    
                    items.append(current_item)
                    
                except Exception as e:
                    # Parsing failed, maybe wrapped line?
                    if current_item:
                        current_item["description"] += " " + line
            elif len(numbers) == 0 and current_item:
                # Continuation of description
                # Ignore if it looks like garbage
                if len(line) > 3:
                     current_item["description"] += " " + line

        return items

    def _extract_tax_summary(self, text: str) -> Dict[str, Any]:
        """Extract tax details"""
        # Look for CGST, SGST, IGST totals
        summary = {
            "taxable_value": 0.0,
            "cgst": {"amount": 0.0, "rate": 0},
            "sgst": {"amount": 0.0, "rate": 0},
            "igst": {"amount": 0.0, "rate": 0},
            "total_tax": 0.0
        }
        
        # Regex for Tax Line: "CGST ... 9.00 ... 450.00"
        lines = text.splitlines()
        for line in lines:
            if "CGST" in line:
               params = re.findall(r"[\d,]+\.\d{2}", line)
               if params: summary["cgst"]["amount"] = float(params[-1].replace(",", ""))
            if "SGST" in line:
               params = re.findall(r"[\d,]+\.\d{2}", line)
               if params: summary["sgst"]["amount"] = float(params[-1].replace(",", ""))
            if "IGST" in line:
               params = re.findall(r"[\d,]+\.\d{2}", line)
               if params: summary["igst"]["amount"] = float(params[-1].replace(",", ""))
        
        summary["total_tax"] = summary["cgst"]["amount"] + summary["sgst"]["amount"] + summary["igst"]["amount"]
        return summary

    def _extract_totals(self, text: str) -> Dict[str, Any]:
        totals = {
            "grand_total": 0.0,
            "amount_words": None,
            "total_tax": 0.0 # Placeholder, usually from tax summary
        }
        
        # Grand Total
        # Look for "Total" followed by biggest number or "Grand Total"
        # Strategy: Find lines with "Total" and currency/number
        lines = text.splitlines()
        for line in lines:
            if "Grand Total" in line or ("Total" in line and "Qty" not in line):
                 nums = re.findall(r"[\d,]+\.\d{2}", line)
                 if nums:
                     # Usually the last number is the Grand Total
                     totals["grand_total"] = float(nums[-1].replace(",", ""))
            
            if "Amount Chargeable (in words)" in line:
                # Might be next line
                # Simple check: extract text after label
                parts = line.split("words)")
                if len(parts) > 1 and parts[1].strip():
                    totals["amount_words"] = parts[1].strip()
                # Or fetch next line in full parser logic (simplified here)

        return totals

    def get_flattened_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert extracting structured data to flat rows for Excel"""
        rows = []
        
        # Prepare Master Info
        master = {}
        if data.get("header_metadata"):
            master["IRN"] = data["header_metadata"].get("irn")
            master["Ack No"] = data["header_metadata"].get("ack_no")
            master["Ack Date"] = data["header_metadata"].get("ack_date")
        
        if data.get("invoice_metadata"):
            master["Invoice No"] = data["invoice_metadata"].get("invoice_no")
            master["Invoice Date"] = data["invoice_metadata"].get("invoice_date")
            master["Dispatched Through"] = data["invoice_metadata"].get("dispatched_through")
            master["Destination"] = data["invoice_metadata"].get("destination")
        
        if data.get("seller_address"):
            master["Seller Name"] = data["seller_address"].get("company_name")
            master["Seller GSTIN"] = data["seller_address"].get("gstin")
            master["Seller Address"] = data["seller_address"].get("address")
        
        if data.get("buyer_bill_to"):
            master["Buyer Name"] = data["buyer_bill_to"].get("company_name")
            master["Buyer GSTIN"] = data["buyer_bill_to"].get("gstin")
            master["Buyer Address"] = data["buyer_bill_to"].get("address")
        
        if data.get("totals"):
            master["Grand Total"] = data["totals"].get("grand_total")
            master["Amount In Words"] = data["totals"].get("amount_words")
            
        if data.get("tax_summary"):
             master["Total Tax"] = data["tax_summary"].get("total_tax")

        # Create rows
        if data.get("line_items"):
            for item in data["line_items"]:
                row = master.copy()
                row.update({
                    "Sl No": item.get("sl_no"),
                    "Description": item.get("description"),
                    "HSN/SAC": item.get("hsn_sac"),
                    "Quantity": item.get("quantity"),
                    "Unit": item.get("unit"),
                    "Rate": item.get("rate"),
                    "Amount": item.get("amount")
                })
                rows.append(row)
        else:
            rows.append(master)
            
        return rows

    def to_excel(self, data: Dict[str, Any], output_path: str):
        try:
            df = pd.DataFrame(self.get_flattened_data(data))
            df.to_excel(output_path, index=False)
            print(f"Saved to {output_path}")
        except Exception as e:
            print(f"Excel save failed: {e}")

if __name__ == "__main__":
    # Test block
    extractor = InvoiceDataExtractor()
    # Mock call if file provided
    # res = extractor.extract("sample.pdf")
    # print(json.dumps(res, indent=2))
