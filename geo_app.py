st.header("Step 2: Multi-Cell Test Builder")
            num_cells = st.number_input("How many separate test cells are you running?", min_value=1, max_value=5, value=1)
            
            halflife_map = {
                "High-Intent DR (Search, Shopping)": 3, 
                "Feed-Based Social (Meta, TikTok)": 7, 
                "Immersive / Lean-Back (CTV, YouTube, TV, Audio)": 14
            }
            lag_map = {"Low (<$50, Impulse)": 1, "Medium ($50-$200)": 7, "High ($200+, Heavy research)": 14}
            
            # --- NEW: OPTIMIZATION TOGGLE ---
            auto_optimize = st.toggle("🤖 Auto-Optimize Cadence & Pairs for Lowest % Lift (MDE)", value=True)
            if auto_optimize:
                st.caption("The engine will simulate all possible combinations to find the exact cadence and number of pairs that mathematically minimizes your target % lift.")

            # --- PHASE A: GATHER ALL CELL SETTINGS FIRST ---
            cell_configs = []
            for i in range(num_cells):
                st.markdown(f"### ⚙️ Settings for Test Cell {i+1}")
                c1, c2, c3, c4 = st.columns(4)
                cell_name = c1.text_input(f"Campaign/Cell Name", f"Campaign {i+1}", key=f"name_{i}")
                
                if not auto_optimize:
                    cadence = c2.selectbox(f"Match Cadence", ["Daily", "Weekly", "Monthly"], key=f"cadence_{i}")
                    num_pairs = c3.number_input(f"Pairs to Auto-Select", 1, 50, 5, key=f"num_{i}")
                else:
                    cadence = None # Will be populated by the optimizer
                    num_pairs = None # Will be populated by the optimizer
                    c2.info("Cadence: Auto")
                    c3.info("Pairs: Auto")
                    
                target_roas = c4.number_input("Target Break-Even ROAS", 0.1, 20.0, 2.0, step=0.1, key=f"roas_{i}")
                
                ac1, ac2 = st.columns(2)
                channel = ac1.selectbox("Media Format & Attention Level", list(halflife_map.keys()), key=f"chan_{i}")
                consideration = ac2.selectbox("Product Price / Consideration", list(lag_map.keys()), key=f"cons_{i}")
                
                cell_configs.append({
                    'id': i, 'name': cell_name, 'cadence': cadence, 'num_pairs': num_pairs,
                    'roas': target_roas, 'channel': channel, 'consideration': consideration
                })

            # --- NEW: OPTIMIZATION ENGINE ---
            if auto_optimize:
                with st.spinner("Simulating combinations to find lowest % Lift..."):
                    best_avg_mde = float('inf')
                    best_cadence = "Daily"
                    best_pairs = 1
                    
                    # Search grid: Test all cadences and pair counts
                    for test_cadence in ["Daily", "Weekly", "Monthly"]:
                        available_df = results_df[results_df['Matched_On'] == test_cadence]
                        max_pairs_possible = len(available_df) // num_cells
                        
                        # We test from 1 pair up to a max of 30 per cell to find the sweet spot
                        for test_pairs in range(1, min(max_pairs_possible + 1, 31)):
                            
                            # 1. Simulate Greedy Assignment
                            sim_assignments = {c['id']: [] for c in cell_configs}
                            sim_volumes = {c['id']: 0 for c in cell_configs}
                            pool = available_df.sort_values(by='T_Volume', ascending=False).head(test_pairs * num_cells)
                            
                            for _, pair in pool.iterrows():
                                eligible_cells = [c['id'] for c in cell_configs if len(sim_assignments[c['id']]) < test_pairs]
                                if not eligible_cells: break
                                target_cell_id = min(eligible_cells, key=lambda x: sim_volumes[x])
                                sim_assignments[target_cell_id].append(pair)
                                sim_volumes[target_cell_id] += pair['T_Volume']
                                
                            # 2. Simulate MDE Calculation
                            combo_mdes = []
                            for config in cell_configs:
                                sim_df = pd.DataFrame(sim_assignments[config['id']])
                                if sim_df.empty: 
                                    combo_mdes.append(float('inf'))
                                    continue
                                    
                                hl_days = halflife_map[config['channel']]
                                lag_days = lag_map[config['consideration']]
                                calc_test_days = max(28, int(np.ceil((lag_days * 2) / 7.0) * 7))
                                
                                t_dmas = sim_df['Treatment_DMA'].tolist()
                                c_dmas = sim_df['Control_DMA'].tolist()
                                t_sum = daily_pivot[t_dmas].sum(axis=1)
                                c_sum = daily_pivot[c_dmas].sum(axis=1)
                                
                                if test_cadence == 'Weekly':
                                    t_sum = t_sum.resample('W-MON').sum()
                                    c_sum = c_sum.resample('W-MON').sum()
                                    periods = calc_test_days / 7.0
                                elif test_cadence == 'Monthly':
                                    t_sum = t_sum.resample('MS').sum()
                                    c_sum = c_sum.resample('MS').sum()
                                    periods = calc_test_days / 30.0
                                else:
                                    periods = calc_test_days
                                    
                                volume_scalar = t_sum.sum() / c_sum.sum() if c_sum.sum() > 0 else 1
                                c_scaled = c_sum * volume_scalar
                                
                                sd_diff = np.std(t_sum - c_scaled)
                                mde_abs = 2.8 * (sd_diff * np.sqrt(periods))
                                base_vol = t_sum.mean() * periods
                                mde_pct = (mde_abs / base_vol) * 100 if base_vol > 0 else float('inf')
                                combo_mdes.append(mde_pct)
                                
                            # 3. Evaluate if this combo is the new best
                            avg_mde = np.mean(combo_mdes)
                            if avg_mde < best_avg_mde:
                                best_avg_mde = avg_mde
                                best_cadence = test_cadence
                                best_pairs = test_pairs
                                
                    # Apply the winning combo to our configs
                    for config in cell_configs:
                        config['cadence'] = best_cadence
                        config['num_pairs'] = best_pairs
                        
                    st.success(f"✨ **Optimization Complete!** To minimize variance, the engine selected a **{best_cadence}** match cadence with **{best_pairs}** pairs per cell. (Predicted Average MDE: {best_avg_mde:.1f}%)")

            # --- PHASE B: GREEDY VOLUME BALANCING (THE ACTUAL RUN) ---
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
