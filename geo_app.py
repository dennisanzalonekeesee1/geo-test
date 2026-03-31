import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go

# 1. Access the password from secrets
VALID_PASSWORD = st.secrets["auth"]["password"]

def check_auth():
    """Returns True if the user has entered the correct password."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔒 Geo-Tool Access Control")
        user_input = st.text_input("Enter Access Code", type="password")
        if st.button("Unlock"):
            if user_input == VALID_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid Code.")
        return False
    return True

# 2. Halt execution if not authenticated
if not check_auth():
    st.stop()

# --- CODE ---
st.success("Access Granted.")
# start...

st.set_page_config(page_title="Geo-Test App", layout="wide")

st.title("📍 Geo-Test App")
st.markdown("Market Matching, Multi-Cell Planning, Post-Test Causal Measurement.")

# --- SIDEBAR: NAVIGATION & UPLOADS ---
with st.sidebar:
    app_mode = st.radio("🔄 Select App Mode", ["1. Pre-Test Planner", "2. Post-Test Measurement"])
    test_direction = st.radio("Test Direction", ["Scale-Up (Ads ON)", "Holdout (Ads OFF)"])
    st.divider()

# --- CACHED DATA PROCESSING (Shared by both modes) ---
@st.cache_data
def load_data(sales_file, zip_dma_file):
    df_sales = pd.read_csv(sales_file) if sales_file.name.endswith('.csv') else pd.read_excel(sales_file)
    df_map = pd.read_csv(zip_dma_file) if zip_dma_file.name.endswith('.csv') else pd.read_excel(zip_dma_file)
    return df_sales, df_map

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
            # NEW: Calculate volume for balancing logic later
            t_vols = daily_pivot.sum()
            results_df['T_Volume'] = results_df['Treatment_DMA'].map(t_vols)

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
            
        st.header("Step 1: The Matchmaker Waterfall")
        if not results_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Pairs Available", len(results_df))
            col2.metric("Daily Pairs", len(results_df[results_df['Matched_On'] == 'Daily']))
            col3.metric("Weekly Pairs", len(results_df[results_df['Matched_On'] == 'Weekly']))
            col4.metric("Monthly Pairs", len(results_df[results_df['Matched_On'] == 'Monthly']))
            
            with st.expander("View All Generated Pairs (The Dating Pool)"):
                st.dataframe(results_df, use_container_width=True)
            
            st.header("Step 2: Multi-Cell Test Builder")
            num_cells = st.number_input("How many separate test cells are you running?", min_value=1, max_value=5, value=1)
            
            halflife_map = {
                "High-Intent DR (Search, Shopping)": 3, 
                "Feed-Based Social (Meta, TikTok)": 7, 
                "Immersive / Lean-Back (CTV, YouTube, TV, Audio)": 14
            }
            lag_map = {"Low (<$50, Impulse)": 1, "Medium ($50-$200)": 7, "High ($200+, Heavy research)": 14}
            
            # --- PHASE A: GATHER ALL CELL SETTINGS FIRST ---
            cell_configs = []
            for i in range(num_cells):
                st.markdown(f"### ⚙️ Settings for Test Cell {i+1}")
                c1, c2, c3, c4 = st.columns(4)
                cell_name = c1.text_input(f"Campaign/Cell Name", f"Campaign {i+1}", key=f"name_{i}")
                cadence = c2.selectbox(f"Match Cadence", ["Daily", "Weekly", "Monthly"], key=f"cadence_{i}")
                num_pairs = c3.number_input(f"Pairs to Auto-Select", 1, 50, 5, key=f"num_{i}")
                target_roas = c4.number_input("Target Break-Even ROAS", 0.1, 20.0, 2.0, step=0.1, key=f"roas_{i}")
                
                ac1, ac2 = st.columns(2)
                channel = ac1.selectbox("Media Format & Attention Level", list(halflife_map.keys()), key=f"chan_{i}")
                consideration = ac2.selectbox("Product Price / Consideration", list(lag_map.keys()), key=f"cons_{i}")
                
                cell_configs.append({
                    'id': i, 'name': cell_name, 'cadence': cadence, 'num_pairs': num_pairs,
                    'roas': target_roas, 'channel': channel, 'consideration': consideration
                })
            
            # --- PHASE B: GREEDY VOLUME BALANCING ---
            assigned_pair_ids = []
            cell_assignments = {i: pd.DataFrame() for i in range(num_cells)}
            
            for current_cadence in ["Daily", "Weekly", "Monthly"]:
                competing_cells = [c for c in cell_configs if c['cadence'] == current_cadence]
                if not competing_cells: continue
                
                total_pairs_needed = sum(c['num_pairs'] for c in competing_cells)
                available_df = results_df[(results_df['Matched_On'] == current_cadence) & (~results_df['Pair_ID'].isin(assigned_pair_ids))]
                
                if total_pairs_needed > len(available_df):
                    st.error(f"Not enough {current_cadence} pairs to fill all requests. You need {total_pairs_needed}, but only {len(available_df)} are available.")
                    st.stop()
                    
                # Sort pool by Volume descending so we deal out the largest markets first
                pool = available_df.sort_values(by='T_Volume', ascending=False).head(total_pairs_needed)
                
                # Track current volume per cell to keep them balanced
                cell_volumes = {c['id']: 0 for c in competing_cells}
                assigned_rows = {c['id']: [] for c in competing_cells}
                
                for _, pair in pool.iterrows():
                    # Check which cells still need pairs
                    eligible_cells = [c['id'] for c in competing_cells if len(assigned_rows[c['id']]) < c['num_pairs']]
                    if not eligible_cells: break
                    
                    # Give the pair to the eligible cell with the LOWEST current total volume
                    target_cell_id = min(eligible_cells, key=lambda x: cell_volumes[x])
                    
                    assigned_rows[target_cell_id].append(pair)
                    cell_volumes[target_cell_id] += pair['T_Volume']
                    assigned_pair_ids.append(pair['Pair_ID'])
                    
                # Convert list of rows back to DataFrames
                for c in competing_cells:
                    cell_assignments[c['id']] = pd.DataFrame(assigned_rows[c['id']])

            st.divider()

            # --- PHASE C: CALCULATE ECONOMICS & DISPLAY ---
            for config in cell_configs:
                i = config['id']
                cell_df = cell_assignments[i]
                cell_name = config['name']
                cadence = config['cadence']
                
                hl_days = halflife_map[config['channel']]
                lag_days = lag_map[config['consideration']]
                
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
                recommended_budget = mde_absolute / config['roas'] if config['roas'] > 0 else 0
                
                st.markdown(f"### 🧪 Results: {cell_name}")
                with st.expander(f"📊 View Economics & Export for: {cell_name}", expanded=True):
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.metric("Active Run Time", f"{calc_test_days} Days")
                    bc2.metric("Adstock Cooldown", f"{calc_cooldown} Days")
                    bc3.metric("Incremental Sales Needed", f"${mde_absolute:,.0f} ({mde_pct:.1f}% Lift)")
                    budget_label = "Required Total Budget" if test_direction == "Scale-Up (Ads ON)" else "Spend to Withhold"
                    bc4.metric(budget_label, f"${recommended_budget:,.0f}")
                    
                    chart_data = pd.DataFrame({'Treatment': t_sum, 'Control (Scaled)': c_scaled}).reset_index()
                    fig = px.line(chart_data, x=date_col, y=['Treatment', 'Control (Scaled)'], title=f"Historical Baseline: {cell_name}", labels={'value':'Gross Sales', 'variable':'Group'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    csv = cell_df.to_csv(index=False).encode('utf-8')
                    st.download_button(f"📥 Download Activation Map: {cell_name}", data=csv, file_name=f'test_cell_{i+1}.csv', mime='text/csv', key=f"dl_{i}")
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
        spend_label = "Actual Media Spend ($)" if test_direction == "Scale-Up (Ads ON)" else "Withheld Media Spend ($)"
        actual_spend = st.number_input(spend_label, min_value=1.0, value=10000.0, step=500.0)
        
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
            cadence = test_map['Matched_On'].iloc[0] # Auto-detect how these pairs were matched!
            
            df_test = df[df[dma_col2].isin(t_dmas + c_dmas)]
            daily_pivot = df_test.pivot_table(index=date_col2, columns=dma_col2, values=sales_col2, aggfunc='sum').fillna(0)
            
            # Ensuring missing DMAs are padded with 0s
            for d in (t_dmas + c_dmas):
                if d not in daily_pivot.columns: daily_pivot[d] = 0
            
            daily_pivot['Treatment_Actual'] = daily_pivot[t_dmas].sum(axis=1)
            daily_pivot['Control_Actual'] = daily_pivot[c_dmas].sum(axis=1)
            
            # Cadence Adjustment for Measurement
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
                # --- SYNTHETIC CONTROL MODEL (Linear Regression) ---
                X_pre = pre_data['Control_Actual'].values
                Y_pre = pre_data['Treatment_Actual'].values
                
                # Fit 1D polynomial (y = mx + b)
                slope, intercept = np.polyfit(X_pre, Y_pre, 1)
                
                # Predicting Counterfactual for the ENTIRE timeline
                model_data['Counterfactual (Predicted)'] = (model_data['Control_Actual'] * slope) + intercept
                model_data['Counterfactual (Predicted)'] = model_data['Counterfactual (Predicted)'].clip(lower=0) 
                
                model_data['Incremental_Lift'] = model_data['Treatment_Actual'] - model_data['Counterfactual (Predicted)']
                
                # --- STATISTICAL MEASUREMENT (Post-Period Only) ---
                post_model = model_data[(model_data.index >= start_dt) & (model_data.index <= end_dt)]
                
                total_treatment = post_model['Treatment_Actual'].sum()
                total_counterfactual = post_model['Counterfactual (Predicted)'].sum()
                incremental_revenue = total_treatment - total_counterfactual
                roas = incremental_revenue / actual_spend if actual_spend > 0 else 0
                
                # 95% Confidence Interval & Standard Error based on pre-period variance
                pre_residuals = pre_data['Treatment_Actual'] - ((pre_data['Control_Actual'] * slope) + intercept)
                sigma = np.std(pre_residuals)
                se_total = sigma * np.sqrt(len(post_model))
                
                ci_lower = incremental_revenue - (1.96 * se_total)
                ci_upper = incremental_revenue + (1.96 * se_total)
                
                # --- DYNAMIC HOLDOUT LOGIC ---
                if test_direction == "Scale-Up (Ads ON)":
                    stat_sig = ci_lower > 0
                    is_success = incremental_revenue > 0
                    display_revenue = incremental_revenue
                    display_roas = roas
                    success_msg = "✅ **STATISTICALLY SIGNIFICANT WIN:** The ads drove proven incremental revenue! (Confidence Interval is entirely above $0)."
                    warn_msg = "⚠️ **NOT SIGNIFICANT / INCONCLUSIVE:** The lift was positive but indistinguishable from natural market variance (noise). The confidence interval includes zero."
                    fail_msg = "🚨 **NEGATIVE OR ZERO LIFT:** The Treatment markets underperformed compared to the mathematical baseline. The ads did not work."
                else: # Holdout (Ads OFF)
                    stat_sig = ci_upper < 0
                    is_success = incremental_revenue < 0 # A drop in sales is a success!
                    display_revenue = abs(incremental_revenue) # Show absolute dollars protected
                    display_roas = abs(incremental_revenue) / actual_spend if actual_spend > 0 else 0
                    success_msg = "✅ **STATISTICALLY SIGNIFICANT HOLDOUT:** Turning ads OFF caused a proven drop in sales! Your baseline spend is highly incremental. (CI is entirely below $0)."
                    warn_msg = "⚠️ **INCONCLUSIVE:** The sales dropped, but it was indistinguishable from natural market variance. The confidence interval includes zero."
                    fail_msg = "🚨 **NO DROP DETECTED:** Turning ads OFF did not cause a meaningful drop in sales. Your baseline spend is likely taking credit for organic sales."

                st.header("Step 1: Test Results & Significance")
                if stat_sig:
                    st.success(success_msg)
                elif is_success:
                    st.warning(warn_msg)
                else:
                    st.error(fail_msg)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Incremental Impact", f"${display_revenue:,.0f}")
                m2.metric("True ROAS", f"{display_roas:.2f}x")
                m3.metric("95% Confidence Interval", f"${ci_lower:,.0f} to ${ci_upper:,.0f}")
                m4.metric("% Lift over Baseline", f"{(incremental_revenue / total_counterfactual)*100:.1f}%" if total_counterfactual > 0 else "N/A")
                
                # --- VISUALIZATIONS ---
                st.header("Step 2: Causal Impact Visualization")
                
                # Chart 1: Time Series (Actual vs Counterfactual)
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=model_data.index, y=model_data['Treatment_Actual'], mode='lines', name='Actual Treatment Sales', line=dict(color='#00b4d8', width=3)))
                fig1.add_trace(go.Scatter(x=model_data.index, y=model_data['Counterfactual (Predicted)'], mode='lines', name='Counterfactual (No Ads)', line=dict(color='#ff9f1c', width=3, dash='dash')))
                
                # Added shaded test window
                fig1.add_vrect(x0=start_dt, x1=end_dt, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Active Test & Cooldown Period", annotation_position="top left")
                
                fig1.update_layout(title=f"Sales Impact: Actual vs. Predicted ({cadence} Level)", yaxis_title="Gross Sales ($)", hovermode="x unified")
                st.plotly_chart(fig1, use_container_width=True)
                
                # Chart 2: Cumulative Lift 
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
