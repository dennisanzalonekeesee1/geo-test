import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px

st.set_page_config(page_title="Geo-Test Matchmaker Pro", layout="wide")

st.title("📍 Geo-Test Matchmaker & Multi-Cell Planner")
st.markdown("Automate DMA matching, auto-allocate pairs to isolated test cells, and calculate channel-specific budgets.")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("1. Upload Data")
    sales_file = st.file_uploader("Upload Shopify Sales", type=["csv", "xlsx"])
    zip_dma_file = st.file_uploader("Upload Zip-to-DMA Dict", type=["csv", "xlsx"])
    
    st.header("2. Match Settings")
    min_corr = st.slider("Target Correlation Threshold", 0.70, 0.99, 0.85, 0.01)
    
    st.markdown("### Verify Column Names")
    date_col = st.text_input("Date Column (Sales)", "Day")
    zip_col = st.text_input("Zip Code Column (Sales)", "Shipping postal code")
    sales_col = st.text_input("Sales Column (Sales)", "Gross sales")
    dma_col = st.text_input("DMA Column (Dictionary)", "dma_description")
    dict_zip_col = st.text_input("Zip Column (Dictionary)", "zip_code")

# --- CACHED DATA PROCESSING ---
@st.cache_data
def load_data(sales_file, zip_dma_file):
    df_sales = pd.read_csv(sales_file) if sales_file.name.endswith('.csv') else pd.read_excel(sales_file)
    df_map = pd.read_csv(zip_dma_file) if zip_dma_file.name.endswith('.csv') else pd.read_excel(zip_dma_file)
    return df_sales, df_map

@st.cache_data
def process_data(df_sales_raw, df_map_raw, date_col, zip_col, sales_col, dma_col, dict_zip_col, min_corr):
    df_sales = df_sales_raw.copy()
    df_map = df_map_raw.copy()
    
    df_sales[date_col] = pd.to_datetime(df_sales[date_col])
    df_sales[sales_col] = pd.to_numeric(df_sales[sales_col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
    df_sales = df_sales[df_sales[sales_col] > 0] 
    
    df_sales['Clean_Zip'] = df_sales[zip_col].astype(str).str.extract(r'(\d{4,5})')[0].str.zfill(5)
    df_map['Clean_Zip'] = df_map[dict_zip_col].astype(str).str.zfill(5)
    df = pd.merge(df_sales, df_map, on='Clean_Zip', how='inner')
    
    dma_totals = df.groupby(dma_col)[sales_col].sum().sort_values(ascending=False)
    
    if len(dma_totals) > 110: 
        valid_dmas = dma_totals.iloc[10:-100].index.tolist()
        trim_msg = f"Started with {len(dma_totals)} DMAs. Removed Top 10 and Bottom 100. **{len(valid_dmas)} DMAs** remain."
        trim_success = True
    else:
        valid_dmas = dma_totals.index.tolist()
        trim_msg = f"Only found {len(dma_totals)} DMAs. Not enough to safely trim."
        trim_success = False
        
    df_filtered = df[df[dma_col].isin(valid_dmas)]
    daily_pivot = df_filtered.pivot_table(index=date_col, columns=dma_col, values=sales_col, aggfunc='sum').fillna(0)
    
    def find_pairs(df_pivot, min_corr):
        corr_matrix = df_pivot.corr()
        corr_matrix.index.name = None
        corr_matrix.columns.name = None
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_pairs = upper_tri.stack().reset_index()
        corr_pairs.columns = ['DMA_1', 'DMA_2', 'Correlation']
        corr_pairs = corr_pairs[corr_pairs['Correlation'] >= min_corr].sort_values('Correlation', ascending=False)
        
        pairs = []
        paired = set() 
        for _, row in corr_pairs.iterrows():
            d1, d2, corr = row['DMA_1'], row['DMA_2'], row['Correlation']
            if d1 not in paired and d2 not in paired:
                roles = random.sample([d1, d2], 2)
                pairs.append({'Treatment_DMA': roles[0], 'Control_DMA': roles[1], 'Correlation': round(corr, 4)})
                paired.update([d1, d2])
        return pairs, paired

    # WATERFALL PASS 1
    daily_pairs, daily_paired_dmas = find_pairs(daily_pivot, min_corr)
    for p in daily_pairs: p['Matched_On'] = 'Daily'
    
    # WATERFALL PASS 2
    leftover_dmas_1 = [d for d in valid_dmas if d not in daily_paired_dmas]
    weekly_pairs = []
    weekly_paired_dmas = set()
    if len(leftover_dmas_1) > 1:
        weekly_pivot = daily_pivot[leftover_dmas_1].resample('W-MON').sum()
        weekly_pairs, weekly_paired_dmas = find_pairs(weekly_pivot, min_corr)
        for p in weekly_pairs: p['Matched_On'] = 'Weekly'

    # WATERFALL PASS 3
    leftover_dmas_2 = [d for d in leftover_dmas_1 if d not in weekly_paired_dmas]
    monthly_pairs = []
    if len(leftover_dmas_2) > 1:
        monthly_pivot = daily_pivot[leftover_dmas_2].resample('MS').sum() 
        monthly_pairs, _ = find_pairs(monthly_pivot, min_corr)
        for p in monthly_pairs: p['Matched_On'] = 'Monthly'

    all_pairs = daily_pairs + weekly_pairs + monthly_pairs
    results_df = pd.DataFrame(all_pairs)
    
    if not results_df.empty:
        results_df.index = results_df.index + 1
        results_df.index.name = 'Pair_ID'
        results_df = results_df.reset_index()
        
    return results_df, daily_pivot, trim_msg, trim_success

# --- MAIN LOGIC ---
if sales_file and zip_dma_file:
    with st.spinner("Processing data through waterfall..."):
        df_sales_raw, df_map_raw = load_data(sales_file, zip_dma_file)
        results_df, daily_pivot, trim_msg, trim_success = process_data(
            df_sales_raw, df_map_raw, date_col, zip_col, sales_col, dma_col, dict_zip_col, min_corr
        )
        
    st.header("Step 1: The Matchmaker Waterfall")
    if not results_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Pairs Available", len(results_df))
        col2.metric("Daily Pairs", len(results_df[results_df['Matched_On'] == 'Daily']))
        col3.metric("Weekly Pairs", len(results_df[results_df['Matched_On'] == 'Weekly']))
        col4.metric("Monthly Pairs", len(results_df[results_df['Matched_On'] == 'Monthly']))
        
        with st.expander("View All Generated Pairs (The Dating Pool)"):
            st.dataframe(results_df, use_container_width=True)
        
        # --- MULTI-CELL AUTO-ALLOCATOR ---
        st.header("Step 2: Multi-Cell Test Builder")
        st.markdown("Build concurrent tests safely. Pairs assigned to Cell 1 are mathematically locked out of Cell 2 to prevent SUTVA violations. Cells are strictly filtered by matching cadence to prevent variance explosions.")
        
        num_cells = st.number_input("How many separate test cells are you running?", min_value=1, max_value=5, value=1)
        
        assigned_pair_ids = [] 
        
        # 🎯 PRECISE INDUSTRY HEURISTICS FOR ADSTOCK & LAG
        halflife_map = {
            "High-Intent DR (Search, Shopping, Retargeting)": 3, 
            "Feed-Based Social (Meta Image/Video, TikTok, Reels)": 7, 
            "Immersive / Lean-Back (CTV, YouTube, TV, Audio)": 14
        }
        lag_map = {
            "Low (<$50, Impulse)": 1, 
            "Medium ($50-$200, Mild consideration)": 7, 
            "High ($200+, Heavy research)": 14
        }
        
        for i in range(num_cells):
            st.markdown(f"### 🧪 Test Cell {i+1}")
            
            c1, c2, c3, c4 = st.columns(4)
            cell_name = c1.text_input(f"Campaign/Cell Name", f"Campaign {i+1}", key=f"name_{i}")
            cadence = c2.selectbox(f"Match Cadence", ["Daily", "Weekly", "Monthly"], key=f"cadence_{i}", help="Strictly isolates pairs by how they matched to protect statistical variance.")
            
            available_df = results_df[(results_df['Matched_On'] == cadence) & (~results_df['Pair_ID'].isin(assigned_pair_ids))]
            max_available = len(available_df)
            
            if max_available == 0:
                st.error(f"0 {cadence} pairs left! They are either all assigned to other cells, or none met the correlation threshold.")
                continue
                
            num_pairs = c3.number_input(f"Pairs to Auto-Select (Max {max_available})", 1, max_available, min(5, max_available), key=f"num_{i}")
            target_roas = c4.number_input("Target Break-Even ROAS", 0.1, 20.0, 2.0, step=0.1, key=f"roas_{i}")
            
            cell_df = available_df.head(num_pairs)
            assigned_pair_ids.extend(cell_df['Pair_ID'].tolist())
            
            ac1, ac2 = st.columns(2)
            channel = ac1.selectbox("Media Format & Attention Level", list(halflife_map.keys()), key=f"chan_{i}", help="Determines the memory retention (Adstock Decay) of your ads.")
            consideration = ac2.selectbox("Product Price / Consideration", list(lag_map.keys()), key=f"cons_{i}")
            
            # --- DYNAMIC POWER ANALYSIS ---
            hl_days = halflife_map[channel]
            lag_days = lag_map[consideration]
            
            calc_cooldown = lag_days + (hl_days * 2)
            calc_test_days = max(28, int(np.ceil((lag_days * 2) / 7.0) * 7))
            
            t_dmas = cell_df['Treatment_DMA'].tolist()
            c_dmas = cell_df['Control_DMA'].tolist()
            
            t_sum = daily_pivot[t_dmas].sum(axis=1)
            c_sum = daily_pivot[c_dmas].sum(axis=1)
            
            if cadence == 'Weekly':
                t_sum = t_sum.resample('W-MON').sum()
                c_sum = c_sum.resample('W-MON').sum()
                periods = calc_test_days / 7.0
            elif cadence == 'Monthly':
                t_sum = t_sum.resample('MS').sum()
                c_sum = c_sum.resample('MS').sum()
                periods = calc_test_days / 30.0
            else:
                periods = calc_test_days
                
            volume_scalar = t_sum.sum() / c_sum.sum() if c_sum.sum() > 0 else 1
            c_scaled = c_sum * volume_scalar
            
            diffs = t_sum - c_scaled
            sd_diff = np.std(diffs)
            
            se_total = sd_diff * np.sqrt(periods) 
            mde_absolute = 2.8 * se_total
            
            baseline_t_vol = t_sum.mean() * periods
            mde_pct = (mde_absolute / baseline_t_vol) * 100 if baseline_t_vol > 0 else 0
            recommended_budget = mde_absolute / target_roas if target_roas > 0 else 0
            
            with st.expander(f"📊 View Economics & Export for: {cell_name} ({num_pairs} Pairs)", expanded=True):
                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric("Active Run Time", f"{calc_test_days} Days")
                bc2.metric("Adstock Cooldown", f"{calc_cooldown} Days")
                bc3.metric("Incremental Sales Needed", f"${mde_absolute:,.0f} ({mde_pct:.1f}% Lift)")
                bc4.metric("Required Total Budget", f"${recommended_budget:,.0f}")
                
                if mde_pct <= 10:
                    st.success("✅ **Highly Feasible:** Your required lift is small. Low risk of ad saturation.")
                elif mde_pct <= 20:
                    st.warning("⚠️ **Moderate Risk:** Ensure strong creative and manage frequency caps.")
                else:
                    st.error("🚨 **High Risk of Saturation:** This cell requires a massive lift to beat the noise. Allocate MORE pairs to this cell to lower the variance!")
                
                chart_data = pd.DataFrame({'Treatment': t_sum, 'Control (Scaled)': c_scaled}).reset_index()
                fig = px.line(chart_data, x=date_col, y=['Treatment', 'Control (Scaled)'], 
                              title=f"Historical Baseline ({cadence}): {cell_name}",
                              labels={'value': 'Gross Sales', 'variable': 'Group'})
                st.plotly_chart(fig, use_container_width=True)
                
                csv = cell_df.to_csv(index=False).encode('utf-8')
                st.download_button(label=f"📥 Download Activation Map: {cell_name}", data=csv, file_name=f'test_cell_{i+1}_{cell_name.replace(" ", "_")}.csv', mime='text/csv')
                
            st.divider()

    else:
        st.error(f"No pairs found. Try lowering the threshold.")
else:
    st.info("👈 Please upload your Shopify Sales and Zip-to-DMA Dictionary in the sidebar to begin.")
