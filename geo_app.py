import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Geo-Test OS", layout="wide")

st.title("📍 Geo-Testing Operating System")
st.markdown("End-to-End Market Matching, Multi-Cell Planning, and Post-Test Causal Measurement.")

# --- SIDEBAR: NAVIGATION & UPLOADS ---
with st.sidebar:
    app_mode = st.radio("🔄 Select App Mode", ["1. Pre-Test Planner", "2. Post-Test Measurement"])
    st.divider()

# --- CACHED DATA PROCESSING (Shared by both modes) ---
@st.cache_data
def load_data(sales_file, zip_dma_file):
    df_sales = pd.read_csv(sales_file) if sales_file.name.endswith('.csv') else pd.read_excel(sales_file)
    df_map = pd.read_csv(zip_dma_file) if zip_dma_file.name.endswith('.csv') else pd.read_excel(zip_dma_file)
    return df_sales, df_map

# Helper to calculate MDE purely for the optimization loop
def calculate_cell_mde(t_dmas, c_dmas, cadence, periods, daily_pivot):
    t_sum = daily_pivot[t_dmas].sum(axis=1)
    c_sum = daily_pivot[c_dmas].sum(axis=1)
    if cadence == 'Weekly':
        t_sum = t_sum.resample('W-MON').sum()
        c_sum = c_sum.resample('W-MON').sum()
    elif cadence == 'Monthly':
        t_sum = t_sum.resample('MS').sum()
        c_sum = c_sum.resample('MS').sum()
        
    vol_scalar = t_sum.sum() / c_sum.sum() if c_sum.sum() > 0 else 1
    c_scaled = c_sum * vol_scalar
    diffs = t_sum - c_scaled
    sd_diff = np.std(diffs)
    se_total = sd_diff * np.sqrt(periods) 
    mde_abs = 2.8 * se_total
    baseline_t_vol = t_sum.mean() * periods
    mde_pct = (mde_abs / baseline_t_vol) * 100 if baseline_t_vol > 0 else float('inf')
    return mde_pct, mde_abs

# ==========================================
# MODE 1: PRE-TEST PLANNER
# ==========================================
if app_mode == "1. Pre-Test Planner":
    with st.sidebar:
        st.header("1. Upload Data")
        sales_file = st.file_uploader("Upload Historical Sales", type=["csv", "xlsx"])
        zip_dma_file = st.file_uploader("Upload Zip-to-DMA Dict", type=["csv", "xlsx"])
        
        st.header("2. Match Settings")
        min_corr = st.slider("Target Correlation Threshold", 0.70, 0.99, 0.85, 0.01)
        
        st.markdown("### Verify Column Names")
        date_col = st.text_input("Date Column (Sales)", "Day")
        zip_col = st.text_input("Zip Code Column (Sales)", "Shipping postal code")
        sales_col = st.text_input("Sales Column (Sales)", "Gross sales")
        dma_col = st.text_input("DMA Column (Dictionary)", "dma_description")
        dict_zip_col = st.text_input("Zip Column (Dictionary)", "zip_code")

    @st.cache_data
    def process_pre_test(df_sales_raw, df_map_raw, date_col, zip_col, sales_col, dma_col, dict_zip_col, min_corr):
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
            trim_msg = f"Started with {len(dma_totals)} DMAs. Removed Top 10 and Bottom 100. **{len(valid_dmas)} DMAs** remain for pairing."
            trim_success = True
        else:
            valid_dmas = dma_totals.index.tolist()
            trim_msg = f"Only found {len(dma_totals)} DMAs. Not enough to safely trim without losing data."
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

        # Waterfall Passes
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

    if sales_file and zip_dma_file:
        with st.spinner("Processing data through waterfall..."):
            df_sales_raw, df_map_raw = load_data(sales_file, zip_dma_file)
            results_df, daily_pivot, trim_msg, trim_success = process_pre_test(
                df_sales_raw, df_map_raw, date_col, zip_col, sales_col, dma_col, dict_zip_col, min_corr
            )
            
        st.header("Step 1: Outlier Trimming & Pairing Results")
        if trim_success:
            st.success(trim_msg)
        else:
            st.warning(trim_msg)

        if not results_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Pairs Available", len(results_df))
            col2.metric("Daily Pairs", len(results_df[results_df['Matched_On'] == 'Daily']))
            col3.metric("Weekly Pairs", len(results_df[results_df['Matched_On'] == 'Weekly']))
            col4.metric("Monthly Pairs", len(results_df[results_df['Matched_On'] == 'Monthly']))
            
            with st.expander("🔍 View All Generated Pairs & Individual Visualizer"):
                st.dataframe(results_df, use_container_width=True)
                
                st.markdown("#### Inspect an Individual Pair")
                pair_options = results_df.apply(lambda x: f"Pair {x['Pair_ID']}: {x['Treatment_DMA']} vs {x['Control_DMA']} ({x['Matched_On']}, r={x['Correlation']})", axis=1).tolist()
                
                if pair_options:
                    selected_pair_str = st.selectbox("Select a pair to verify their historical trends move together:", pair_options)
                    selected_idx = pair_options.index(selected_pair_str)
                    t_dma = results_df.iloc[selected_idx]['Treatment_DMA']
                    c_dma = results_df.iloc[selected_idx]['Control_DMA']
                    matched_on = results_df.iloc[selected_idx]['Matched_On']
                    
                    chart_data_ind = daily_pivot[[t_dma, c_dma]]
                    if matched_on == 'Weekly': chart_data_ind = chart_data_ind.resample('W-MON').sum()
                    elif matched_on == 'Monthly': chart_data_ind = chart_data_ind.resample('MS').sum()
                        
                    fig_ind = px.line(chart_data_ind.reset_index(), x=date_col, y=[t_dma, c_dma], title=f"Historical Match ({matched_on})", labels={'value': 'Gross Sales', 'variable': 'Assignment'})
                    st.plotly_chart(fig_ind, use_container_width=True)

            st.header("Step 2: Multi-Cell Test Builder & Math Optimizer")
            st.markdown("Build concurrent tests safely. **The script mathematically evaluates every combination of Cadences and Pair Counts to guarantee the lowest-risk scenario, automatically adapting based on how many cells you request.**")
            
            num_cells = st.number_input("How many separate test cells are you running concurrently?", min_value=1, max_value=20, value=1)
            
            assigned_pair_ids = [] 
            halflife_map = {
                "High-Intent DR (Search, Shopping, Retargeting)": 3, 
                "Feed-Based Social (Meta Image/Video, TikTok, Reels)": 7, 
                "Immersive / Lean-Back (CTV, YouTube, TV, Audio)": 14
            }
            lag_map = {"Low (<$50, Impulse)": 1, "Medium ($50-$200, Mild consideration)": 7, "High ($200+, Heavy research)": 14}
            
            for i in range(num_cells):
                st.markdown(f"### 🧪 Test Cell {i+1}")
                remaining_cells = num_cells - 1 - i
                total_avail_all = len(results_df[~results_df['Pair_ID'].isin(assigned_pair_ids)])
                
                if total_avail_all <= remaining_cells and remaining_cells > 0:
                    st.error("Not enough total pairs remaining to fulfill all test cells. Please reduce your cell count or reduce pairs assigned to previous cells.")
                    break
                elif total_avail_all == 0:
                    st.error("No pairs left!")
                    break
                
                # Row 1: Define the Campaign FIRST so we can calculate the test duration properly
                c1, c2, c3, c4 = st.columns(4)
                cell_name = c1.text_input(f"Campaign Name", f"Campaign {i+1}", key=f"name_{i}")
                channel = c2.selectbox("Media Format & Attention Level", list(halflife_map.keys()), key=f"chan_{i}")
                consideration = c3.selectbox("Product Price / Consideration", list(lag_map.keys()), key=f"cons_{i}")
                target_roas = c4.number_input("Target Break-Even ROAS", 0.1, 20.0, 2.0, step=0.1, key=f"roas_{i}")
                
                # Math Params based on Campaign selection
                hl_days = halflife_map[channel]
                lag_days = lag_map[consideration]
                calc_test_days = max(28, int(np.ceil((lag_days * 2) / 7.0) * 7))
                calc_cooldown = lag_days + (hl_days * 2)
                
                # --- OPTIMIZATION LOOP (Find lowest risk setup) ---
                best_cadence = "Daily"
                best_k = 1
                lowest_mde = float('inf')
                
                available_cadences = []
                
                for cad in ["Daily", "Weekly", "Monthly"]:
                    df_cad = results_df[(results_df['Matched_On'] == cad) & (~results_df['Pair_ID'].isin(assigned_pair_ids))]
                    if len(df_cad) > 0:
                        available_cadences.append(cad)
                        
                        max_k = min(len(df_cad), total_avail_all - remaining_cells) if remaining_cells > 0 else len(df_cad)
                        
                        if max_k >= 1:
                            if cad == 'Weekly': periods = calc_test_days / 7.0
                            elif cad == 'Monthly': periods = calc_test_days / 30.0
                            else: periods = calc_test_days
                            
                            for k in range(1, max_k + 1):
                                t_dmas_tmp = df_cad.head(k)['Treatment_DMA'].tolist()
                                c_dmas_tmp = df_cad.head(k)['Control_DMA'].tolist()
                                mde_pct_tmp, _ = calculate_cell_mde(t_dmas_tmp, c_dmas_tmp, cad, periods, daily_pivot)
                                
                                if mde_pct_tmp < lowest_mde:
                                    lowest_mde = mde_pct_tmp
                                    best_cadence = cad
                                    best_k = k

                # Row 2: Auto-Allocated Cadence and Pairs
                st.markdown("#### 🎯 Footprint Auto-Allocator")
                r2c1, r2c2 = st.columns(2)
                
                if not available_cadences:
                    st.error("No valid cadences left.")
                    break
                    
                cadence_idx = available_cadences.index(best_cadence) if best_cadence in available_cadences else 0
                selected_cadence = r2c1.selectbox(f"Match Cadence", available_cadences, index=cadence_idx, key=f"cadence_{i}")
                
                # Recalculate if user manually overrides the optimized cadence
                df_selected_cad = results_df[(results_df['Matched_On'] == selected_cadence) & (~results_df['Pair_ID'].isin(assigned_pair_ids))]
                max_k_selected = min(len(df_selected_cad), total_avail_all - remaining_cells) if remaining_cells > 0 else len(df_selected_cad)
                max_k_selected = max(1, max_k_selected)
                
                if selected_cadence == best_cadence:
                    default_k = best_k
                else:
                    default_k = 1
                    lowest_mde_manual = float('inf')
                    if selected_cadence == 'Weekly': periods_sel = calc_test_days / 7.0
                    elif selected_cadence == 'Monthly': periods_sel = calc_test_days / 30.0
                    else: periods_sel = calc_test_days
                    
                    for k in range(1, max_k_selected + 1):
                        t_dmas_tmp = df_selected_cad.head(k)['Treatment_DMA'].tolist()
                        c_dmas_tmp = df_selected_cad.head(k)['Control_DMA'].tolist()
                        mde_pct_tmp, _ = calculate_cell_mde(t_dmas_tmp, c_dmas_tmp, selected_cadence, periods_sel, daily_pivot)
                        if mde_pct_tmp < lowest_mde_manual:
                            lowest_mde_manual = mde_pct_tmp
                            default_k = k
                            
                num_pairs = r2c2.number_input(f"Pairs (Max {max_k_selected})", min_value=1, max_value=max_k_selected, value=default_k, key=f"num_{i}")
                
                if selected_cadence == best_cadence and num_pairs == best_k:
                    st.success(f"✨ **Mathematically Optimized Default:** Evaluated every combination and automatically selected **{selected_cadence} Cadence** with **{num_pairs} Pairs** to achieve the absolute lowest risk scenario (**{lowest_mde:.1f}% Lift Required**).")
                else:
                    st.info(f"🔧 **Manual Override Active:** You shifted away from the lowest-risk mathematical default. Watch the Diminishing Returns check below!")

                # Lock them in for SUTVA
                cell_df = df_selected_cad.head(num_pairs)
                assigned_pair_ids.extend(cell_df['Pair_ID'].tolist())
                
                # Final Stats for the Expander
                if selected_cadence == 'Weekly': periods_final = calc_test_days / 7.0
                elif selected_cadence == 'Monthly': periods_final = calc_test_days / 30.0
                else: periods_final = calc_test_days
                
                mde_pct_final, mde_abs_final = calculate_cell_mde(cell_df['Treatment_DMA'].tolist(), cell_df['Control_DMA'].tolist(), selected_cadence, periods_final, daily_pivot)
                recommended_budget = mde_abs_final / target_roas if target_roas > 0 else 0
                
                with st.expander(f"📊 View Economics & Export for: {cell_name} ({num_pairs} Pairs)", expanded=True):
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.metric("Active Run Time", f"{calc_test_days} Days")
                    bc2.metric("Adstock Cooldown", f"{calc_cooldown} Days")
                    bc3.metric("Incremental Sales Needed", f"${mde_abs_final:,.0f} ({mde_pct_final:.1f}% Lift)")
                    bc4.metric("Required Total Budget", f"${recommended_budget:,.0f}")
                    
                    st.markdown("### Diminishing Returns Reality Check")
                    if mde_pct_final <= 10:
                        st.success(f"✅ **Highly Feasible (Requires {mde_pct_final:.1f}% Lift):** Safe to execute for these {num_pairs} pairs. Low risk of ad saturation.")
                    elif mde_pct_final <= 20:
                        st.warning(f"⚠️ **Moderate Risk (Requires {mde_pct_final:.1f}% Lift):** You need a sizable lift. Ensure strong creative and manage frequency caps.")
                    else:
                        st.error(f"🚨 **High Risk of Saturation (Requires {mde_pct_final:.1f}% Lift):** The historical noise is too high compared to the volume of the markets you selected. Trying to force this lift will likely cause ad fatigue before you hit statistical significance.")
                    
                    t_sum_f = daily_pivot[cell_df['Treatment_DMA'].tolist()].sum(axis=1)
                    c_sum_f = daily_pivot[cell_df['Control_DMA'].tolist()].sum(axis=1)
                    if selected_cadence == 'Weekly':
                        t_sum_f = t_sum_f.resample('W-MON').sum()
                        c_sum_f = c_sum_f.resample('W-MON').sum()
                    elif selected_cadence == 'Monthly':
                        t_sum_f = t_sum_f.resample('MS').sum()
                        c_sum_f = c_sum_f.resample('MS').sum()
                    
                    vol_scalar_f = t_sum_f.sum() / c_sum_f.sum() if c_sum_f.sum() > 0 else 1
                    c_scaled_f = c_sum_f * vol_scalar_f
                    
                    chart_data = pd.DataFrame({'Treatment': t_sum_f, 'Control (Scaled)': c_scaled_f}).reset_index()
                    fig = px.line(chart_data, x=date_col, y=['Treatment', 'Control (Scaled)'], title=f"Historical Baseline: {cell_name}", labels={'value':'Gross Sales', 'variable':'Group'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    csv = cell_df.to_csv(index=False).encode('utf-8')
                    st.download_button(f"📥 Download Activation Map: {cell_name}", data=csv, file_name=f'test_cell_{i+1}.csv', mime='text/csv')
                st.divider()
        else:
            st.error(f"No pairs found. Try lowering the threshold.")
    else:
        st.info("👈 Please upload your Historical Sales and Zip Dictionary in the sidebar to begin planning.")


# ==========================================
# MODE 2: POST-TEST MEASUREMENT
# ==========================================
elif app_mode == "2. Post-Test Measurement":
    with st.sidebar:
        st.header("1. Upload Data")
        post_sales_file = st.file_uploader("Upload Full Sales Data (Pre + Post Test)", type=["csv", "xlsx"], help="Must include history before the test to train the model!")
        zip_dma_file_post = st.file_uploader("Upload Zip-to-DMA Dict", type=["csv", "xlsx"])
        test_map_file = st.file_uploader("Upload Test Cell Map (CSV from Planner)", type=["csv"])
        
        st.header("2. Campaign Details")
        start_date = st.date_input("Test Start Date (Ads turned ON)")
        end_date = st.date_input("Measurement End Date (End of Cooldown)")
        actual_spend = st.number_input("Actual Media Spend ($)", min_value=1.0, value=10000.0, step=500.0)
        
        st.markdown("### Verify Column Names")
        date_col2 = st.text_input("Date Column (Sales)", "Day", key="d2")
        zip_col2 = st.text_input("Zip Code Column (Sales)", "Shipping postal code", key="z2")
        sales_col2 = st.text_input("Sales Column (Sales)", "Gross sales", key="s2")
        dma_col2 = st.text_input("DMA Column (Dictionary)", "dma_description", key="dm2")
        dict_zip_col2 = st.text_input("Zip Column (Dictionary)", "zip_code", key="dz2")

    st.title("📈 Post-Test Causal Measurement")
    st.markdown("Calculate true incremental lift and ROAS using a Time-Series Synthetic Control regression model.")

    if post_sales_file and zip_dma_file_post and test_map_file:
        with st.spinner("Training Synthetic Control Model..."):
            df_sales_raw, df_map_raw = load_data(post_sales_file, zip_dma_file_post)
            test_map = pd.read_csv(test_map_file)
            
            df_sales = df_sales_raw.copy()
            df_map = df_map_raw.copy()
            
            df_sales[date_col2] = pd.to_datetime(df_sales[date_col2])
            df_sales[sales_col2] = pd.to_numeric(df_sales[sales_col2].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            df_sales['Clean_Zip'] = df_sales[zip_col2].astype(str).str.extract(r'(\d{4,5})')[0].str.zfill(5)
            df_map['Clean_Zip'] = df_map[dict_zip_col2].astype(str).str.zfill(5)
            df = pd.merge(df_sales, df_map, on='Clean_Zip', how='inner')
            
            t_dmas = test_map['Treatment_DMA'].tolist()
            c_dmas = test_map['Control_DMA'].tolist()
            cadence = test_map['Matched_On'].iloc[0] 
            
            df_test = df[df[dma_col2].isin(t_dmas + c_dmas)]
            daily_pivot = df_test.pivot_table(index=date_col2, columns=dma_col2, values=sales_col2, aggfunc='sum').fillna(0)
            
            for d in (t_dmas + c_dmas):
                if d not in daily_pivot.columns: daily_pivot[d] = 0
            
            daily_pivot['Treatment_Actual'] = daily_pivot[t_dmas].sum(axis=1)
            daily_pivot['Control_Actual'] = daily_pivot[c_dmas].sum(axis=1)
            
            if cadence == 'Weekly':
                model_data = daily_pivot[['Treatment_Actual', 'Control_Actual']].resample('W-MON').sum()
            elif cadence == 'Monthly':
                model_data = daily_pivot[['Treatment_Actual', 'Control_Actual']].resample('MS').sum()
            else:
                model_data = daily_pivot[['Treatment_Actual', 'Control_Actual']].copy()
            
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            pre_data = model_data[model_data.index < start_dt]
            post_data = model_data[(model_data.index >= start_dt) & (model_data.index <= end_dt)]
            
            if len(pre_data) < 3:
                st.error("🚨 Not enough pre-test data to train the model. Your uploaded file MUST include historical sales from before the Test Start Date.")
            elif len(post_data) == 0:
                st.error("🚨 No data found for the Test Period. Check your start/end dates.")
            else:
                X_pre = pre_data['Control_Actual'].values
                Y_pre = pre_data['Treatment_Actual'].values
                
                slope, intercept = np.polyfit(X_pre, Y_pre, 1)
                
                model_data['Counterfactual (Predicted)'] = (model_data['Control_Actual'] * slope) + intercept
                model_data['Counterfactual (Predicted)'] = model_data['Counterfactual (Predicted)'].clip(lower=0) 
                
                model_data['Incremental_Lift'] = model_data['Treatment_Actual'] - model_data['Counterfactual (Predicted)']
                
                post_model = model_data[(model_data.index >= start_dt) & (model_data.index <= end_dt)]
                
                total_treatment = post_model['Treatment_Actual'].sum()
                total_counterfactual = post_model['Counterfactual (Predicted)'].sum()
                incremental_revenue = total_treatment - total_counterfactual
                roas = incremental_revenue / actual_spend if actual_spend > 0 else 0
                
                pre_residuals = pre_data['Treatment_Actual'] - ((pre_data['Control_Actual'] * slope) + intercept)
                sigma = np.std(pre_residuals)
                se_total = sigma * np.sqrt(len(post_model))
                
                ci_lower = incremental_revenue - (1.96 * se_total)
                ci_upper = incremental_revenue + (1.96 * se_total)
                stat_sig = ci_lower > 0
                
                st.header("Step 1: Test Results & Significance")
                if stat_sig:
                    st.success("✅ **STATISTICALLY SIGNIFICANT:** The campaign successfully pierced the market noise and drove proven incremental revenue! (The 95% Confidence Interval is entirely above $0).")
                elif incremental_revenue > 0:
                    st.warning("⚠️ **NOT SIGNIFICANT / INCONCLUSIVE:** The incremental lift was positive but indistinguishable from natural market variance (noise). The confidence interval includes zero.")
                else:
                    st.error("🚨 **NEGATIVE OR ZERO LIFT:** The Treatment markets underperformed compared to the mathematical baseline. The ads did not work.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Incremental Revenue", f"${incremental_revenue:,.0f}")
                m2.metric("Realized ROAS", f"{roas:.2f}x")
                m3.metric("95% Confidence Interval", f"${ci_lower:,.0f} to ${ci_upper:,.0f}")
                m4.metric("% Lift over Baseline", f"{(incremental_revenue / total_counterfactual)*100:.1f}%" if total_counterfactual > 0 else "N/A")
                
                st.header("Step 2: Causal Impact Visualization")
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=model_data.index, y=model_data['Treatment_Actual'], mode='lines', name='Actual Treatment Sales', line=dict(color='#00b4d8', width=3)))
                fig1.add_trace(go.Scatter(x=model_data.index, y=model_data['Counterfactual (Predicted)'], mode='lines', name='Counterfactual (No Ads)', line=dict(color='#ff9f1c', width=3, dash='dash')))
                fig1.add_vrect(x0=start_dt, x1=end_dt, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Active Test & Cooldown Period", annotation_position="top left")
                fig1.update_layout(title=f"Sales Impact: Actual vs. Predicted ({cadence} Level)", yaxis_title="Gross Sales ($)", hovermode="x unified")
                st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("### Cumulative Incremental Lift")
                st.markdown("This shows the incremental dollars piling up over time during the test. The green shaded area represents the 95% Confidence Interval. If the lower boundary of the green area stays above the red zero-line by the end of the test, it's a statistically significant win!")
                
                post_model_cum = post_model.copy()
                post_model_cum['Cumulative_Lift'] = post_model_cum['Incremental_Lift'].cumsum()
                post_model_cum['CI_Upper'] = post_model_cum['Cumulative_Lift'] + (1.96 * sigma * np.sqrt(np.arange(1, len(post_model)+1)))
                post_model_cum['CI_Lower'] = post_model_cum['Cumulative_Lift'] - (1.96 * sigma * np.sqrt(np.arange(1, len(post_model)+1)))

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=post_model_cum.index, y=post_model_cum['Cumulative_Lift'], mode='lines', name='Cumulative Incremental Sales', line=dict(color='green', width=3)))
                fig2.add_trace(go.Scatter(x=post_model_cum.index, y=post_model_cum['CI_Upper'], mode='lines', line=dict(width=0), showlegend=False))
                fig2.add_trace(go.Scatter(x=post_model_cum.index, y=post_model_cum['CI_Lower'], mode='lines', fill='tonexty', fillcolor='rgba(0,128,0,0.2)', line=dict(width=0), name='95% Confidence Interval'))
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                fig2.update_layout(title="Cumulative ROI Over Time", yaxis_title="Cumulative Incremental Sales ($)", hovermode="x unified")
                st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("👈 Please upload your Post-Test Sales, Zip Dictionary, and the Test Cell CSV generated during Phase 1.")
