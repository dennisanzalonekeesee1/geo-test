import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px

st.set_page_config(page_title="Geo-Test Matchmaker Pro", layout="wide")

st.title("📍 Geo-Test Matchmaker & Pre-Test Planner")
st.markdown("Automate DMA matching, test cell selection, adstock cooldowns, and budget power analysis.")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("1. Upload Data")
    sales_file = st.file_uploader("Upload Shopify Sales", type=["csv", "xlsx"])
    zip_dma_file = st.file_uploader("Upload Zip-to-DMA Dict", type=["csv", "xlsx"])
    
    st.header("2. Match Settings")
    min_corr = st.slider("Target Correlation Threshold", 0.70, 0.99, 0.85, 0.01)
    
    st.header("3. Adstock & Cooldown")
    ad_channel = st.selectbox("Media Channel (Decay Rate)", [
        "Search / Bottom Funnel (Fast Decay)",
        "Social / Mid Funnel (Medium Decay)",
        "Video / CTV / Audio (Slow Decay)"
    ])
    consideration = st.selectbox("Product Consideration (Purchase Lag)", [
        "Low (Impulse, <$50)",
        "Medium ($50 - $200)",
        "High (Research, $200+)"
    ])
    
    st.header("4. Business Economics")
    expected_roas = st.number_input(
        "Target (Break-Even) ROAS", 
        min_value=0.1, max_value=20.0, value=2.0, step=0.1,
        help="Minimum ROAS needed for success. We power the budget to detect this."
    )
    
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

    daily_pairs, daily_paired_dmas = find_pairs(daily_pivot, min_corr)
    for p in daily_pairs: p['Matched_On'] = 'Daily'
    
    leftover_dmas_1 = [d for d in valid_dmas if d not in daily_paired_dmas]
    weekly_pairs = []
    weekly_paired_dmas = set()
    if len(leftover_dmas_1) > 1:
        weekly_pivot = daily_pivot[leftover_dmas_1].resample('W-MON').sum()
        weekly_pairs, weekly_paired_dmas = find_pairs(weekly_pivot, min_corr)
        for p in weekly_pairs: p['Matched_On'] = 'Weekly'

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
        
    st.header("Step 1: Correlation & Pairing Results")
    if not results_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Pairs", len(results_df))
        col2.metric("Matched Daily", len(results_df[results_df['Matched_On'] == 'Daily']))
        col3.metric("Matched Weekly", len(results_df[results_df['Matched_On'] == 'Weekly']))
        col4.metric("Matched Monthly", len(results_df[results_df['Matched_On'] == 'Monthly']))
        
        st.dataframe(results_df, use_container_width=True)
        
        # --- NEW MULTI-CELL BUILDER ---
        st.header("Step 2: Build Your Test Cell (Multi-Cell Builder)")
        st.markdown("Select a specific block of pairs for this test cell to isolate the budget calculations. *Tip: You can use Pairs 1-5 for one test, and Pairs 6-10 for a concurrent, separate test without breaking SUTVA.*")
        
        # Multi-select UI
        pair_strings = results_df.apply(lambda x: f"Pair {x['Pair_ID']} ({x['Matched_On']}, r={x['Correlation']}): {x['Treatment_DMA']} vs {x['Control_DMA']}", axis=1).tolist()
        
        # By default, pre-select ONLY Daily pairs to protect the math!
        default_pairs = results_df[results_df['Matched_On'] == 'Daily'].apply(
            lambda x: f"Pair {x['Pair_ID']} ({x['Matched_On']}, r={x['Correlation']}): {x['Treatment_DMA']} vs {x['Control_DMA']}", axis=1
        ).tolist()
        
        selected_pair_strings = st.multiselect("Select Pairs to include in this Test Cell:", options=pair_strings, default=default_pairs)
        
        if selected_pair_strings:
            selected_ids = [int(p.split(" ")[1]) for p in selected_pair_strings]
            test_df = results_df[results_df['Pair_ID'].isin(selected_ids)]
            
            # Warning about mixing cadences
            cadences = test_df['Matched_On'].unique()
            if len(cadences) > 1:
                st.warning("⚠️ **Methodology Warning:** You selected a mix of Daily, Weekly, or Monthly pairs. Because they have different historical variance levels, measuring them together in a daily time-series model post-test will be noisy. **Recommendation:** Only group pairs from the same matching tier.")

            # --- DYNAMIC BUDGET & POWER CALCULATION ---
            st.header("Step 3: Power Analysis & Budget Math")
            
            # Adstock / Cooldown Math based on industry heuristics
            halflife_map = {"Search / Bottom Funnel (Fast Decay)": 3, "Social / Mid Funnel (Medium Decay)": 7, "Video / CTV / Audio (Slow Decay)": 14}
            lag_map = {"Low (Impulse, <$50)": 1, "Medium ($50 - $200)": 7, "High (Research, $200+)": 14}
            
            hl_days = halflife_map[ad_channel]
            lag_days = lag_map[consideration]
            
            # Cooldown = Product Lag + (2 * Half-life to capture ~75%+ of decay)
            calc_cooldown = lag_days + (hl_days * 2)
            
            # Test Length rule of thumb: At least 28 days, or 2x the consideration lag
            calc_test_days = max(28, int(np.ceil((lag_days * 2) / 7.0) * 7))
            
            # Math only runs on the selected pairs!
            t_dmas = test_df['Treatment_DMA'].tolist()
            c_dmas = test_df['Control_DMA'].tolist()
            
            t_daily_sum = daily_pivot[t_dmas].sum(axis=1)
            c_daily_sum = daily_pivot[c_dmas].sum(axis=1)
            
            volume_scalar = t_daily_sum.sum() / c_daily_sum.sum() if c_daily_sum.sum() > 0 else 1
            c_daily_scaled = c_daily_sum * volume_scalar
            
            daily_diffs = t_daily_sum - c_daily_scaled
            sd_diff = np.std(daily_diffs)
            
            se_total = sd_diff * np.sqrt(calc_test_days)
            mde_absolute = 2.8 * se_total
            
            baseline_t_volume = t_daily_sum.mean() * calc_test_days
            mde_pct = (mde_absolute / baseline_t_volume) * 100 if baseline_t_volume > 0 else 0
            recommended_budget = mde_absolute / expected_roas if expected_roas > 0 else 0
            
            st.info(f"💡 **Dynamic Math:** Based on standard heuristics for **{ad_channel}** and **{consideration}** products, we estimate a combined adstock/purchase cooldown of **{calc_cooldown} days**. The budget is calculated *only* for the {len(test_df)} pairs you selected.")
            
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Active Test Length", f"{calc_test_days} Days")
            b_col2.metric("Decay Cooldown", f"{calc_cooldown} Days")
            b_col3.metric("Required Incremental Sales", f"${mde_absolute:,.0f}")
            b_col4.metric(f"Total Budget (For these {len(test_df)} pairs)", f"${recommended_budget:,.0f}")
            
            st.markdown("### Diminishing Returns Reality Check")
            if mde_pct <= 10:
                st.success(f"✅ **Highly Feasible (Requires {mde_pct:.1f}% Lift):** Safe to execute for these {len(test_df)} markets.")
            elif mde_pct <= 20:
                st.warning(f"⚠️ **Moderate Risk (Requires {mde_pct:.1f}% Lift):** Watch frequency caps to avoid ad saturation.")
            else:
                st.error(f"🚨 **High Risk of Diminishing Returns (Requires {mde_pct:.1f}% Lift):** The historical noise is too high compared to the volume of the markets you selected. **Recommendation:** Add MORE pairs to your Test Cell Builder above to increase your baseline volume and lower the required lift percentage.")
                
            # --- VISUAL VALIDATION (Dynamic) ---
            st.header("Step 4: Visual Validation")
            st.markdown("Visualizing the combined sales volume of your selected Treatment footprint vs. your selected Control footprint. Notice how pooling them smooths out the variance!")
            
            chart_data = pd.DataFrame({'Treatment Cell': t_daily_sum, 'Control Cell (Scaled)': c_daily_scaled}).reset_index()
            # If all selected are weekly/monthly, adjust plot
            if all(m == 'Weekly' for m in cadences):
                chart_data = chart_data.set_index(date_col).resample('W-MON').sum().reset_index()
            elif all(m == 'Monthly' for m in cadences):
                chart_data = chart_data.set_index(date_col).resample('MS').sum().reset_index()
                
            fig = px.line(chart_data, x=date_col, y=['Treatment Cell', 'Control Cell (Scaled)'], 
                          title=f"Historical Aggregate: Selected Treatment vs Control Markets",
                          labels={'value': 'Gross Sales', 'variable': 'Group'})
            st.plotly_chart(fig, use_container_width=True)

            # Export
            st.header("Step 5: Export Test Design")
            csv = test_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download This Test Cell (CSV)", data=csv, file_name='geo_test_cell.csv', mime='text/csv')
        else:
            st.warning("Please select at least one pair to calculate the budget.")
    else:
        st.error(f"No pairs found. Try lowering the threshold.")
else:
    st.info("👈 Please upload your Shopify Sales and Zip-to-DMA Dictionary in the sidebar to begin.")
