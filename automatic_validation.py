import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import io
import os
import tempfile
import zipfile
from Extractor import InvoiceDataExtractor
from huggingface_hub import InferenceClient
import pdfplumber
import traceback
from manual_validator import ManualDataValidator
from excel_converter import ExcelToJSONConverter

# --- LLM PROMPT HELPER ---
def get_enhanced_prompt(invoice_text):
    return f"""You are an advanced invoice extraction agent.
EXTRACT EVERY SINGLE PIECE OF DATA from the invoice text below.

STRICT JSON OUTPUT RULES:
1. Output MUST be a valid single JSON object.
2. If a field is found, extract its value exactly as string or number.
3. If a field is NOT found, set its value to null (do not omit it, do not use "Not Found" or empty string, use null).
4. Keys must be snake_case.

MANDATORY FIELDS TO LOOK FOR (include these in output at minimum, plus any others you find):
- invoice_number (Look for 'Invoice No', 'Inv No'. It often starts with 'TS-' and contains year like '/25-26'. Do NOT confuse with address numbers like '472/5')
- invoice_date (Look for 'Dated', 'Invoice Date')
- irn (Invoice Reference Number)
- ack_no (Acknowledgement Number)
- ack_date
- total_amount (Grand Total)
- total_tax_amount
- seller_name
- seller_address
- seller_gstin
- seller_state
- buyer_name
- buyer_address
- buyer_gstin
- buyer_state
- consignee_name
- consignee_address
- consignee_gstin
- dispatch_details (Vehicle No, Dispatch Doc No, etc.)
- line_items (Array of objects: sl_no, description, hsn_sac, quantity, rate, amount, batch_no)

INVOICE TEXT:
<<<
{invoice_text}
>>>

JSON OUTPUT:
"""

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(
    page_title="AI Validator & Extractor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main {
        background-color: #fce4ec; /* Light pink background from streamlit_app.py */
    }
    .stButton>button {
        background-color: #e91e63;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #d81b60;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stDataFrame {
        font-size: 14px;
    }
    .match-cell {
        background-color: #d4edda !important;
    }
    .mismatch-cell {
        background-color: #f8d7da !important;
    }
    .missing-cell {
        background-color: #fff3cd !important;
    }
    .master-only-cell {
        background-color: #d1ecf1 !important;
    }
    .error-cell {
        background-color: #f5c6cb !important;
        color: #721c24 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------ CACHED RESOURCES ------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_openai_client(api_key):
    if api_key:
        return OpenAI(api_key=api_key)
    return None

# ------------------ AI VALIDATOR LOGIC ------------------
class AIDataValidator:
    def __init__(self, client, unique_key="Bill_No"):
        self.unique_key = unique_key
        self.client = client
    
    def validate_with_gpt(self, master_row, check_row, model_name="gpt-4o-mini", max_retries=2):
        """
        Use OpenAI GPT to compare and validate two data rows.
        """
        # Prepare data for AI comparison
        def normalize_for_ai(val):
            """
            Recursively Clean data for AI:
            1. Remove 'item_' prefix from keys in dicts
            2. Convert numbers to strings to avoid type confusion in JSON
            """
            try:
                if isinstance(val, dict):
                    return {k.replace('item_', '').replace('item', '').strip('_'): normalize_for_ai(v) for k, v in val.items()}
                
                # Check for list-likes (list, tuple, numpy array)
                if isinstance(val, (list, tuple)) or hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
                    return [normalize_for_ai(x) for x in val]
                
                # Scalar checks
                if pd.isna(val):
                    return "NULL" 
                
                return str(val)
            except Exception:
                # Fallback for weird types
                return str(val)

        # Apply normalization to create clean dicts for Prompt
        master_clean = normalize_for_ai({k: v for k, v in master_row.items() if k != self.unique_key})
        check_clean = normalize_for_ai({k: v for k, v in check_row.items() if k != self.unique_key})
        
        prompt = f"""
You are an expert data validation AI. Your task is to compare two data records and determine if they match.

**STRICT COMPARISON RULES:**
1. **IGNORE STRUCTURE**: If the content matches but keys are slightly different (e.g., "sl_no" vs "item_sl_no"), it is a MATCH.
2. **IGNORE FORMAT**: Case, punctuation, spacing, and date formats (e.g., "2023-01-01" vs "1-Jan-23") do NOT matter. "100.00" equals "100".
3. **IDENTICAL VALUES**: If two values look strictly identical (e.g. "12345" and "12345"), they ARE A MATCH. Do not hallucinate differences.
4. **ARRAYS/LISTS**: Compare lists item-by-item semantically. If all items match, the list matches.
5. **NULLS**: "NULL" matches "None", "", or missing.

**MASTER RECORD:**
{json.dumps(master_clean, indent=2)}

**CHECKING RECORD:**
{json.dumps(check_clean, indent=2)}

**RETURN FORMAT (JSON only):**
{{
  "match": "YES", "NO", or "ERROR",
  "confidence": "HIGH", "MEDIUM", or "LOW",
  "explanation": "Brief explanation",
  "differences": {{ "field_name": {{ "master_value": "...", "check_value": "...", "analysis": "..." }} }},
  "summary": "Summary string"
}}
"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a data validation expert. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"DEBUG_AI_ERROR: {e}") # Print error to terminal
                if attempt == max_retries - 1:
                    return {"match": "ERROR", "confidence": "LOW", "explanation": f"AI Validation failed: {e}", "differences": {}}
                continue
        return {"match": "ERROR"}

def calculate_similarity_score(master_row, check_row, embedder):
    try:
        master_text = " ".join([f"{k}:{v}" for k, v in master_row.items() if pd.notna(v)])
        check_text = " ".join([f"{k}:{v}" for k, v in check_row.items() if pd.notna(v)])
        master_embed = embedder.encode(master_text)
        check_embed = embedder.encode(check_text)
        similarity = np.dot(master_embed, check_embed) / (np.linalg.norm(master_embed) * np.linalg.norm(check_embed))
        return (similarity + 1) / 2
    except:
        return 0.5

# ------------------ PAGES ------------------

# Helper to clean extracted data types
def clean_extracted_data(data):
    """
    Recursively traverse JSON data and convert string numbers to actual numbers.
    Strings like "1,200.50" -> 1200.5 (float)
    Strings like "50" -> 50 (int)
    """
    if isinstance(data, dict):
        return {k: clean_extracted_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_extracted_data(item) for item in data]
    elif isinstance(data, str):
        # Attempt to convert to number
        # Remove commas
        clean_s = data.replace(',', '').strip()
        try:
            if '.' in clean_s:
                return float(clean_s)
            else:
                # Try int, but some ID-like strings might be caught. 
                # Ideally we only do this for fields that look like math numbers or are known keys.
                # But user asked for "number should be number". 
                # Let's be aggressive but careful with leading zeros (zip codes, etc)
                if clean_s.isdigit():
                    if len(clean_s) > 1 and clean_s.startswith('0'):
                        return data # Keep as string if leading zero
                    return int(clean_s)
                return data
        except ValueError:
            return data
    else:
        return data

def extraction_page(hf_api_key=None):
    st.header("📄 Bulk Invoice Extraction")
    st.info("Upload multiple PDF invoices to extract data. The result will be automatically used as **Master Data** for validation.")
    
    uploaded_files = st.file_uploader("Upload PDF Invoices", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"selected {len(uploaded_files)} files")
        
        if st.button("🚀 Process & Load as Master"):
            # Setup Hugging Face Client
            # Token from UI input or environment variable
            hf_token = hf_api_key if hf_api_key else os.environ.get("HF_TOKEN")
            repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            
            try:
                client = InferenceClient(token=hf_token)
            except Exception as e:
                st.error(f"Failed to initialize Hugging Face CLient: {e}")
                client = None

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_invoice_rows = []
            all_invoice_json_objects = [] # New list for JSON objects
            file_summaries = []
            has_data = False
            
            output_zip = io.BytesIO()
            with zipfile.ZipFile(output_zip, "w") as zf:
                for i, uploaded_file in enumerate(uploaded_files):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name} (AI Extraction)...")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # 1. Extract Text
                        text = ""
                        with pdfplumber.open(tmp_path) as pdf:
                            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                        
                        if not text:
                            st.warning(f"Could not extract text from {uploaded_file.name}")
                            continue

                        # 2. Call LLM
                        if client:
                            full_prompt = get_enhanced_prompt(text)
                            messages = [{"role": "user", "content": full_prompt}]
                            
                            response_text = ""
                            for token in client.chat_completion(
                                messages,
                                model=repo_id, 
                                max_tokens=2500, 
                                stream=True,
                                temperature=0.1
                            ):
                                if token.choices and token.choices[0].delta.content:
                                    response_text += token.choices[0].delta.content
                            
                            # Clean JSON
                            clean_json = response_text.strip()
                            if "```json" in clean_json:
                                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                            elif "```" in clean_json:
                                clean_json = clean_json.split("```")[1].split("```")[0].strip()
                                
                            data = json.loads(clean_json)
                            # Enforce numeric types
                            data = clean_extracted_data(data)
                            
                            # 3. Process Data
                            if data:
                                has_data = True
                                
                                # FLATTEN LOGIC (Recursive for full data caption)
                                base_data = {}
                                
                                # Recursive flattening helper nested here (or could be outer scope)
                                def flatten(y, name_prefix=''):
                                    out = {}
                                    if isinstance(y, dict):
                                        for k, v in y.items():
                                            p = f"{name_prefix}{k}_" if name_prefix else f"{k}_"
                                            out.update(flatten(v, p))
                                    elif isinstance(y, list):
                                        pass # Handle lists separately (line items)
                                    else:
                                        out[name_prefix[:-1] if name_prefix else name_prefix] = y
                                    return out

                                # Flatten top level (excluding line items)
                                for k, v in data.items():
                                    if k != "line_items":
                                        if isinstance(v, dict):
                                            flat_v = flatten(v, f"{k}_")
                                            base_data.update(flat_v)
                                        else:
                                            base_data[k] = v
                                            
                                items = data.get("line_items", [])
                                if isinstance(items, list) and items:
                                    for item in items:
                                        row = base_data.copy()
                                        if isinstance(item, dict):
                                            flat_item = flatten(item, "item_")
                                            row.update(flat_item)
                                        all_invoice_rows.append(row)
                                else:
                                    all_invoice_rows.append(base_data)

                                # Store raw JSON object for nested validation
                                all_invoice_json_objects.append(data)

                                # Summary
                                file_summaries.append({
                                    "File Name": uploaded_file.name,
                                    "Invoice No": data.get("invoice_number"),
                                    "Grand Total": data.get("total_amount")
                                })

                                # Add JSON to zip
                                zf.writestr(f"Invoice_{os.path.splitext(uploaded_file.name)[0]}.json", json.dumps(data, indent=2))
                            
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        traceback.print_exc()
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
            
            progress_bar.empty()
            status_text.success("Processing Complete!")
            
            if has_data and all_invoice_rows:
                df_all = pd.DataFrame(all_invoice_rows)
                
                # Store in Session State
                st.session_state['master_df'] = df_all
                st.session_state['master_json_data'] = all_invoice_json_objects # Store JSON
                st.session_state['master_source'] = "Extracted Invoices"
                
                # NEW: Store display data in session state for persistence
                st.session_state['extraction_display_needed'] = True
                st.session_state['extraction_summaries'] = file_summaries
                
                # Prepare Excel Buffer
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df_all.to_excel(writer, index=False, sheet_name='Consolidated')
                st.session_state['extraction_excel_data'] = excel_buffer.getvalue()

            else:
                st.warning("No data extracted.")

        # OUTSIDE THE BUTTON: Render if data exists in session state
        if st.session_state.get('extraction_display_needed'):
             st.subheader("📊 Extraction Summary")
             st.dataframe(pd.DataFrame(st.session_state['extraction_summaries']), use_container_width=True)
             
             if 'extraction_excel_data' in st.session_state:
                st.download_button(
                    label="📥 Download Consolidated Excel",
                    data=st.session_state['extraction_excel_data'],
                    file_name="Consolidated_Invoices.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
             
             st.success("✅ Data loaded as Master Data! Switch to the 'Validate Data' tab to proceed.")

def validation_page(api_key):
    # --- SIDEBAR (Copied from streamlit_app.py logic) ---
    with st.sidebar:
        st.title("⚙️ Validation Config")
        # API Keys moved to main()

        
        st.divider()
        st.subheader("AI Model Settings")
        model_name = st.selectbox("Model", 
                                 ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"], 
                                 index=0)
        
        st.divider()
        st.subheader("Validation Settings")
        unique_key = st.text_input("Unique Key Column", value="Invoice No")
        
        use_similarity = st.checkbox("Enable Semantic Similarity", value=True,
                                     help="Use embeddings to pre-filter similar records")
        
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05,
                                         help="Rows with similarity below this threshold will skip AI validation")
        
        # Placeholder for dynamic column selector
        column_selection_container = st.container()
        
        st.divider()
        st.info("""
        **AI-Powered Validation Features:**
        1. GPT-based intelligent comparison
        2. Semantic understanding of values
        3. Context-aware matching
        4. Detailed difference analysis
        """)

    # --- MAIN CONTENT ---
    st.title("🤖 AI-Powered Data Validator")
    st.write("Compare **Master Data** vs **Checking Data** using OpenAI GPT for intelligent validation.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📂 Master File")
        master_data = [] # List of dicts
        
        # If extracted data exists in session, offer use of it
        use_extracted = False
        if 'master_json_data' in st.session_state:
             st.success(f"✅ Loaded Extracted Data ({len(st.session_state['master_json_data'])} records)")
             if st.checkbox("Use Extracted Data", value=True):
                 master_data = st.session_state['master_json_data']
                 use_extracted = True
        
        if not use_extracted:
            master_file = st.file_uploader("Upload Master File (Excel/JSON)", type=["xlsx", "xls", "json"], key="master")
            if master_file:
                if master_file.name.endswith('.json'):
                    master_data = json.load(master_file)
                else:
                    # Convert Excel to JSON structure
                    df_m = pd.read_excel(master_file)
                    st.info(f"Converting Master Excel to JSON based on '{unique_key}'...")
                    converter = ExcelToJSONConverter()
                    try:
                        json_str = converter.convert(df_m, unique_key)
                        master_data = json.loads(json_str)
                    except Exception as e:
                        st.error(f"Conversion failed: {e}")

    with col2:
        st.subheader("📂 Checking File")
        check_file = st.file_uploader("Upload Check File (Excel/JSON)", type=["xlsx", "xls", "json"], key="check")
        check_data = []
        if check_file:
            if check_file.name.endswith('.json'):
                check_data = json.load(check_file)
            else:
                df_c = pd.read_excel(check_file)
                st.info(f"Converting Check Excel to JSON based on '{unique_key}'...")
                
                # Use Master Data structure as template if available
                template_item = master_data[0] if master_data else None
                if template_item:
                    st.caption("ℹ️ Using Master Data structure to guide conversion (detecting Line Items).")
                
                converter = ExcelToJSONConverter()
                try:
                    json_str = converter.convert(df_c, unique_key, template_item=template_item)
                    check_data = json.loads(json_str)
                except Exception as e:
                    st.error(f"Conversion failed: {e}")

    if master_data and check_data:
        # Data Preview Tabs
        st.divider()
        tab1, tab2 = st.tabs(["📄 Master Data Preview", "📄 Check Data Preview"])
        with tab1:
            st.json(master_data[:3] if len(master_data) > 3 else master_data)
            st.caption(f"**Master Data:** {len(master_data)} records (Showing first 3)")
            
        with tab2:
            st.json(check_data[:3] if len(check_data) > 3 else check_data)
            st.caption(f"**Check Data:** {len(check_data)} records (Showing first 3)")


        # Identify Common Keys for Selection (from first record)
        master_keys = list(master_data[0].keys()) if master_data else []
        check_keys = list(check_data[0].keys()) if check_data else []
        common_columns = [col for col in master_keys if col in check_keys and col != unique_key]
        
        # Column Selector in Sidebar Container
        with column_selection_container:
            st.divider()
            st.subheader("🎯 Field Selection")
            selected_columns = st.multiselect(
                "Select Fields to Validate",
                options=common_columns,
                default=common_columns,
                help="Select which fields the AI/Manual validator should compare."
            )
            
            if not selected_columns:
                st.warning("⚠️ No fields selected! AI will only check for existence.")

        # Validation Trigger
        st.divider()
        if st.button("🚀 Run AI Validation", use_container_width=True):
            
            if not api_key:
                st.error("❌ Please enter your OpenAI API Key in the sidebar.")
                return
            
            if unique_key not in master_keys or unique_key not in check_keys:
                st.error(f"❌ Unique Key '{unique_key}' not found in both files.")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.write("**Master Fields:**")
                    st.write(master_keys)
                with col_m2:
                    st.write("**Check Fields:**")
                    st.write(check_keys)
            else:
                # Initialize AI validator and embedder
                client = get_openai_client(api_key)
                if not client:
                    st.error("❌ Failed to initialize OpenAI client. Check your API key.")
                    return
                
                validator = AIDataValidator(client, unique_key)
                embedder = load_embedder() if use_similarity else None
                
                results = []
                ai_analysis_details = []  # Store detailed AI analysis
                progress_bar = st.progress(0)
                status_text = st.empty()
                cost_estimator = st.empty()

                checked_keys = set()
                # Initialize Manual Validator
                manual_validator = ManualDataValidator()
                
                # Create Lookup Map for Master Data using Normalized Key
                master_lookup = {}
                for item in master_data:
                    k = item.get(unique_key)
                    if k:
                        norm_k = manual_validator.normalize_text(k)
                        master_lookup[norm_k] = item
                
                total_rows = len(check_data)
                ai_calls = 0
                skipped_by_similarity = 0
                
                # Estimate cost
                avg_tokens_per_call = 800
                cost_per_1k = 0.00015 # gpt-4o-mini default
                if "gpt-4" in model_name and "mini" not in model_name: cost_per_1k = 0.01
                elif "gpt-3.5" in model_name: cost_per_1k = 0.0015
                
                estimated_tokens = total_rows * avg_tokens_per_call
                estimated_cost_val = (estimated_tokens / 1000) * cost_per_1k
                
                with st.expander("📊 Cost Estimation", expanded=False):
                    st.info(f"Estimated cost: ${estimated_cost_val:.4f} for {total_rows} rows")
                
                
                # Process checking file objects
                for i, row_check in enumerate(check_data):
                    progress = (i + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"🤖 Validating record {i+1}/{total_rows}...")
                    
                    key_val = row_check.get(unique_key)
                    checked_keys.add(key_val)
                    
                    # Find corresponding record in Master using robust normalized lookup
                    norm_key_val = manual_validator.normalize_text(key_val)
                    row_master = master_lookup.get(norm_key_val)
                    
                    if not row_master:
                        results.append({
                            "Key": key_val,
                            "Status": "MISSING",
                            "Confidence": "HIGH",
                            "AI Analysis": "Record exists in Checking file but NOT in Master file",
                            "Difference Count": 0,
                            "Details": json.dumps({"error": "Record not found in master"}, indent=2),
                            "AI Calls": 0,
                            "Similarity Score": "N/A"
                        })
                        continue
                    
                    if use_similarity:
                        # --- MANUAL VALIDATION ---
                        # Filter to selected fields
                        row_master_filtered = {k: v for k, v in row_master.items() if k in selected_columns or k == 'line_items'}
                        row_check_filtered = {k: v for k, v in row_check.items() if k in selected_columns or k == 'line_items'}
                        
                        # Call Manual Validator
                        ai_res = manual_validator.validate(row_master_filtered, row_check_filtered)
                        
                        if ai_res.get("match") == "YES":
                            status = "MATCH"
                        elif ai_res.get("match") == "NO":
                            status = "MISMATCH"
                        else:
                            status = "ERROR"
                        
                        confidence = ai_res.get("confidence", "HIGH")
                        explanation = ai_res.get("explanation", "Manual Validation")
                        details = ai_res.get("differences", {})
                        similarity_score = "Manual"
                        skip_ai = True 
                        
                        ai_analysis_details.append({
                            "Key": key_val,
                            "Status": status,
                            "Master Data": row_master,
                            "Check Data": row_check,
                            "AI Response": ai_res,
                            "Similarity Score": similarity_score
                        })
                        
                    else:
                        # --- AI VALIDATION ---
                        ai_calls += 1
                        skip_ai = False
                        similarity_score = "AI Mode"
                        
                        row_master_filtered = {k: v for k, v in row_master.items() if k in selected_columns or k == 'line_items'}
                        row_check_filtered = {k: v for k, v in row_check.items() if k in selected_columns or k == 'line_items'}
                        
                        ai_res = validator.validate_with_gpt(row_master_filtered, row_check_filtered, model_name)
                        
                        current_cost = (ai_calls * avg_tokens_per_call / 1000) * cost_per_1k
                        cost_estimator.text(f"💰 Estimated cost so far: ${current_cost:.4f}")
                        
                        if ai_res.get("match") == "YES":
                            status = "MATCH"
                        elif ai_res.get("match") == "NO":
                            status = "MISMATCH"
                        else:
                            status = "ERROR"
                        
                        confidence = ai_res.get("confidence", "MEDIUM")
                        explanation = ai_res.get("explanation", "No explanation provided")
                        details = ai_res.get("differences", {})
                        
                        ai_analysis_details.append({
                            "Key": key_val,
                            "Status": status,
                            "Master Data": row_master, 
                            "Check Data": row_check,   
                            "AI Response": ai_res,
                            "Similarity Score": similarity_score
                        })
                    
                    difference_count = len(details) if details else 0
                    
                    results.append({
                        "Key": key_val,
                        "Status": status,
                        "Confidence": confidence,
                        "AI Analysis": explanation,
                        "Difference Count": difference_count,
                        "Details": json.dumps(details, indent=2),
                        "AI Calls": 0 if skip_ai else 1,
                        "Similarity Score": f"{similarity_score:.2%}" if not isinstance(similarity_score, str) else similarity_score
                    })
                
                # Check rows that exist in Master but NOT in Checking 
                # missing_in_check = 0 # Simplified
                
                progress_bar.empty()
                status_text.empty()
                cost_estimator.empty()
                
                # --- RESULTS ---
                res_df = pd.DataFrame(results)
                
                # Summary Metrics
                st.divider()
                st.subheader("📊 AI Validation Summary")
                
                m1, m2, m3, m4, m5 = st.columns(5)
                
                total = len(res_df)
                matches = len(res_df[res_df["Status"] == "MATCH"])
                mismatches = len(res_df[res_df["Status"] == "MISMATCH"])
                missing = len(res_df[res_df["Status"].isin(["MISSING", "MASTER_ONLY"])])
                errors = len(res_df[res_df["Status"] == "ERROR"])
                
                m1.metric("Total Records", total)
                m2.metric("✅ AI Matches", matches, f"{matches/total*100 if total else 0:.1f}%")
                m3.metric("❌ Mismatches", mismatches, f"{mismatches/total*100 if total else 0:.1f}%")
                m4.metric("⚠️ Missing/Extra", missing, f"{missing/total*100 if total else 0:.1f}%")
                m5.metric("🤖 AI Calls", ai_calls, f"{skipped_by_similarity} skipped")
                
                # Cost summary
                actual_cost = (ai_calls * avg_tokens_per_call / 1000) * cost_per_1k
                st.info(f"**Cost Summary:** ${actual_cost:.4f} | AI Calls: {ai_calls} | Similarity-filtered: {skipped_by_similarity}")
                
                # Visualization (Donut Chart)
                if not res_df.empty:
                    fig = px.pie(res_df, names="Status", title="AI Validation Results", 
                                 hole=0.4, 
                                 color_discrete_map={
                                     "MATCH": "#28a745",
                                     "MISMATCH": "#dc3545",
                                     "MISSING": "#fd7e14",
                                     "MASTER_ONLY": "#17a2b8",
                                     "ERROR": "#6c757d"
                                 })
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confidence Distribution
                if "Confidence" in res_df.columns:
                    conf_fig = px.histogram(res_df, x="Confidence", 
                                           title="Confidence Level Distribution",
                                           color="Confidence",
                                           color_discrete_map={
                                               "HIGH": "#28a745",
                                               "MEDIUM": "#ffc107",
                                               "LOW": "#dc3545"
                                           })
                    st.plotly_chart(conf_fig, use_container_width=True)
                
                # Detailed AI Analysis & Export
                st.divider()
                st.subheader("📋 Advanced Analysis & Reports")
                
                tab_analysis, tab_export = st.tabs(["🔍 Detailed View", "💾 Export Reports"])
                
                with tab_analysis:
                    if ai_analysis_details:
                        for analysis in ai_analysis_details[:10]:  # Show first 10 for performance
                            with st.expander(f"{analysis['Status']} | Key: {analysis['Key']} | Conf: {analysis['AI Response'].get('confidence', 'N/A')}", expanded=False):
                                col_d1, col_d2 = st.columns(2)
                                with col_d1:
                                    st.caption("Master Data")
                                    st.json(analysis["Master Data"])
                                with col_d2:
                                    st.caption("Checking Data")
                                    st.json(analysis["Check Data"])
                                
                                st.markdown("---")
                                st.markdown(f"**🤖 AI Analysis:** {analysis['AI Response'].get('explanation', 'No explanation')}")
                                if analysis['AI Response'].get('differences'):
                                    st.warning("Found Differences:")
                                    st.json(analysis['AI Response']['differences'])
                    else:
                        st.info("No AI analysis performed yet.")
                
                # Main Results Table
                def highlight_status(val):
                    if val == 'MATCH':
                        return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                    elif val == 'MISMATCH':
                        return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                    elif val == 'MISSING':
                        return 'background-color: #fce5cd; color: #e67700; font-weight: bold;'
                    elif val == 'MASTER_ONLY':
                        return 'background-color: #d1ecf1; color: #0c5460; font-weight: bold;'
                    elif val == 'ERROR':
                        return 'background-color: #f5c6cb; color: #721c24; font-weight: bold;'
                    return ''
                
                def highlight_confidence(val):
                    if val == 'HIGH':
                        return 'background-color: #d4edda; color: #155724;'
                    elif val == 'MEDIUM':
                        return 'background-color: #fff3cd; color: #856404;'
                    elif val == 'LOW':
                        return 'background-color: #f8d7da; color: #721c24;'
                    return ''
                
                # Format display dataframe
                display_df = res_df.copy()
                
                styled_df = display_df.style\
                    .applymap(highlight_status, subset=['Status'])\
                    .applymap(highlight_confidence, subset=['Confidence'])
                
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                with tab_export:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Standard CSV
                        csv = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Summary CSV",
                            data=csv,
                            file_name='ai_validation_summary.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                    
                    with col2:
                        # Prepare Detailed Excel Report
                        if ai_analysis_details:
                            detailed_rows = []
                            for item in ai_analysis_details:
                                flat_row = {
                                    "Unique Key": item["Key"],
                                    "Status": item["Status"],
                                    "Similarity": item.get("Similarity Score", 0),
                                    "AI Explanation": item["AI Response"].get("explanation"),
                                    "Confidence": item["AI Response"].get("confidence")
                                }
                                # Add Master cols
                                for k, v in item["Master Data"].items():
                                    flat_row[f"Master_{k}"] = v
                                # Add Check cols
                                for k, v in item["Check Data"].items():
                                    flat_row[f"Check_{k}"] = v
                                
                                detailed_rows.append(flat_row)
                            
                            df_detailed = pd.DataFrame(detailed_rows)
                            
                            excel_buffer_det = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer_det, engine='xlsxwriter') as writer:
                                df_detailed.to_excel(writer, index=False, sheet_name='Detailed Analysis')
                            
                            st.download_button(
                                label="📥 Download Detailed Analysis (Excel)",
                                data=excel_buffer_det.getvalue(),
                                file_name='detailed_ai_analysis.xlsx',
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            
                            # JSON Download
                            json_detailed = json.dumps(ai_analysis_details, indent=2)
                            st.download_button(
                                label="📄 Download Detailed Analysis (JSON)",
                                data=json_detailed,
                                file_name='detailed_ai_analysis.json',
                                mime='application/json',
                                use_container_width=True
                            )
                        else:
                            st.write("No details to export.")
                
                st.success(f"✅ AI Validation Completed! {ai_calls} AI calls made. ${actual_cost:.4f} estimated cost.")

def json_conversion_page():
    st.header("🔄 Excel to JSON Converter")
    st.info("Convert complex Excel data (like Invoices with multiple line items) into structured JSON.")
    
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            all_cols = list(df.columns)
            unique_key = st.selectbox("Select Unique Key (e.g., Invoice No)", options=all_cols)
            
            if st.button("🚀 Convert to JSON"):
                converter = ExcelToJSONConverter()
                try:
                    json_str = converter.convert(df, unique_key)
                    
                    st.success("✅ Conversion Successful!")
                    st.json(json.loads(json_str)[:3]) # Preview first 3 objects
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_converted.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Conversion failed: {e}")
                    traceback.print_exc()
                    
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ------------------ MAIN APP SHELL ------------------
def main():
    st.sidebar.title("🔍 AiApp")
    
    # --- GLOBAL SIDEBAR ---
    with st.sidebar:
        st.subheader("🔑 Global API Keys")
        openai_api_key = st.text_input("OpenAI API Key", type="password", help="Required for AI Validation")
        hf_api_key_input = st.text_input("Hugging Face API Key", type="password", help="Optional: For Invoice Extraction (overrides env var)")
        st.divider()

    page = st.sidebar.radio("Navigate", ["1. Extract Invoices", "2. Validate Data", "3. Convert to JSON"])
    
    if page == "1. Extract Invoices":
        extraction_page(hf_api_key_input)
    elif page == "2. Validate Data":
        validation_page(openai_api_key)
    elif page == "3. Convert to JSON":
        json_conversion_page()

# Force reload triggers by changing this line
if __name__ == "__main__":
    main()
