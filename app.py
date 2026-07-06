import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Churn Sentinel - Telecom Retention Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI/UX
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Font family overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Custom background color */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Premium Header Container */
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 35px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .header-container h1 {
        color: #ffffff !important;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .header-container p {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-top: 8px;
        margin-bottom: 0;
        font-weight: 400;
    }
    
    /* Card Styles */
    .custom-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #f1f5f9;
        margin-bottom: 25px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Metrics panel */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #f1f5f9 !important;
    }
    
    /* Buttons override */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
        transition: all 0.2s ease;
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
        box-shadow: 0 6px 15px rgba(79, 70, 229, 0.35);
        transform: translateY(-1px);
    }
    
    /* Alert cards custom classes */
    .alert-card-success {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .alert-card-danger {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    /* Styled lists for recommendation cards */
    .rec-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 12px;
        gap: 10px;
        font-size: 0.95rem;
    }
    .rec-icon {
        color: #4f46e5;
        margin-top: 3px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Loading ML Pipeline and Metadata
@st.cache_resource
def load_assets():
    try:
        pipeline = joblib.load("best_model.pkl")
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)
        return pipeline, metadata
    except Exception as e:
        st.error(f"Error loading pipeline or metadata: {e}. Please ensure train_model.py has been run.")
        return None, None

pipeline, metadata = load_assets()

# Function to bin Revenue_Segment based on quantiles
def get_revenue_segment(charges, bins):
    if charges <= bins[1]:
        return "Low"
    elif charges <= bins[2]:
        return "Medium"
    else:
        return "High"

# 3. App Header Banner
st.markdown("""
    <div class="header-container">
        <h1>🔮 Churn Sentinel</h1>
        <p>Advanced predictive intelligence platform for telecom customer retention. Analyze customer behaviors, predict churn risks, and view retention recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

# If assets loaded successfully, render the tabs
if pipeline is not None and metadata is not None:
    tab1, tab2, tab3 = st.tabs([
        "🔮 Customer Risk Analyzer",
        "📊 Model Insights & EDA",
        "📥 Batch Prediction Port"
    ])
    
    # ------------------ TAB 1: CUSTOMER RISK ANALYZER ------------------
    with tab1:
        st.subheader("Compute Customer Churn Risk")
        
        # We group inputs into sidebar sections
        st.sidebar.markdown("## Customer Profile inputs")
        
        # Section A: Account Info
        with st.sidebar.expander("👤 Account & Location", expanded=True):
            state = st.selectbox("State Code", options=metadata["states"], index=metadata["states"].index("KS") if "KS" in metadata["states"] else 0)
            account_length = st.slider("Account Length (Months/Days)", min_value=1, max_value=250, value=100)
            intl_plan = st.selectbox("International Plan", options=["No", "Yes"], index=0)
            vmail_plan = st.selectbox("Voice Mail Plan", options=["No", "Yes"], index=0)
            
        # Section B: Calling Activity
        with st.sidebar.expander("📞 Usage & Calling Volume", expanded=True):
            day_mins = st.slider("Day Call Minutes", min_value=0.0, max_value=400.0, value=180.0)
            day_calls = st.slider("Day Calls Count", min_value=0, max_value=200, value=100)
            
            eve_mins = st.slider("Evening Call Minutes", min_value=0.0, max_value=400.0, value=200.0)
            eve_calls = st.slider("Evening Calls Count", min_value=0, max_value=200, value=100)
            
            night_mins = st.slider("Night Call Minutes", min_value=0.0, max_value=400.0, value=200.0)
            night_calls = st.slider("Night Calls Count", min_value=0, max_value=200, value=100)
            
            intl_mins = st.slider("International Minutes", min_value=0.0, max_value=25.0, value=10.0)
            intl_calls = st.slider("International Calls Count", min_value=0, max_value=20, value=4)

        # Section C: Service Interaction
        with st.sidebar.expander("🛠️ Customer Support Interaction", expanded=True):
            cust_service_calls = st.slider("Customer Service Calls", min_value=0, max_value=10, value=1)
            
        # Perform feature construction & preprocessing
        # Charging rates from telecom standard dataset
        day_charge = day_mins * 0.17
        eve_charge = eve_mins * 0.085
        night_charge = night_mins * 0.045
        intl_charge = intl_mins * 0.27
        total_charges = day_charge + eve_charge + night_charge + intl_charge
        
        total_usage = day_mins + eve_mins + night_mins + intl_mins
        service_stress = cust_service_calls / (account_length + 1)
        
        revenue_segment = get_revenue_segment(total_charges, metadata["revenue_bins"])
        
        # Build Raw Dataframe
        raw_input_df = pd.DataFrame({
            'State': [state],
            'Account length': [account_length],
            'International plan': [intl_plan],
            'Voice mail plan': [vmail_plan],
            'Total day minutes': [day_mins],
            'Total day calls': [day_calls],
            'Total eve minutes': [eve_mins],
            'Total eve calls': [eve_calls],
            'Total night minutes': [night_mins],
            'Total night calls': [night_calls],
            'Total intl minutes': [intl_mins],
            'Total intl calls': [intl_calls],
            'Customer service calls': [cust_service_calls],
            'Total Charges': [total_charges],
            'Total_Usage': [total_usage],
            'Service_Stress': [service_stress],
            'Revenue_Segment': [revenue_segment]
        })
        
        # Process the raw input for model prediction
        model_input_df = raw_input_df.copy()
        
        # Map categorical binary
        model_input_df['International plan'] = model_input_df['International plan'].map({'No': 0, 'Yes': 1})
        model_input_df['Voice mail plan'] = model_input_df['Voice mail plan'].map({'No': 0, 'Yes': 1})
        
        # One-hot encode State and Revenue_Segment
        model_input_df = pd.get_dummies(model_input_df, columns=['State', 'Revenue_Segment'], drop_first=True, dtype=int)
        
        # Reindex features to match X_train exactly (filling dummy columns not present in single prediction with 0)
        model_input_df = model_input_df.reindex(columns=metadata["features"], fill_value=0)
        
        # Run prediction
        pred = pipeline.predict(model_input_df)
        prob = pipeline.predict_proba(model_input_df)
        churn_risk_percentage = prob[0][1] * 100
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
                <div class="custom-card">
                    <div class="card-title">🔍 Predictive Assessment</div>
                """, unsafe_allow_html=True)
            
            # Show visual status
            if pred[0] == 1:
                st.markdown(f"""
                    <div class="alert-card-danger">
                        <h3 style="color:#b91c1c;margin:0 0 10px 0;">⚠️ HIGH CHURN RISK DETECTED</h3>
                        <p style="color:#7f1d1d;margin:0;">The customer shows behavioral indicators strongly correlated with churn. Immediate engagement recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="alert-card-success">
                        <h3 style="color:#047857;margin:0 0 10px 0;">✅ LOW CHURN RISK</h3>
                        <p style="color:#064e3b;margin:0;">The customer exhibits high loyalty signs. Continue normal retention and lifecycle marketing plans.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Gauge Indicator
            fig, ax = plt.subplots(figsize=(6, 2.5), facecolor='none')
            ax.set_facecolor('none')
            
            # Gradient bar
            colors = ["#10b981", "#fbbf24", "#ef4444"]
            cmap = sns.blend_palette(colors, as_cmap=True)
            
            # Draw colorbar
            norm = plt.Normalize(0, 100)
            cb = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax,
                orientation='horizontal',
                shrink=0.9
            )
            cb.outline.set_visible(False)
            cb.set_ticks([0, 25, 50, 75, 100])
            cb.ax.set_xticklabels(['Loyal', 'Low Risk', 'Moderate', 'High Risk', 'Critical'], color='#475569', fontsize=9)
            
            # Add indicator line
            ax.axvline(churn_risk_percentage, color='#0f172a', linewidth=4, ymin=-0.5, ymax=1.5)
            ax.plot(churn_risk_percentage, 0.5, marker='v', color='#0f172a', markersize=14)
            
            plt.title(f"Churn Risk Probability: {churn_risk_percentage:.1f}%", fontsize=12, fontweight='bold', color='#0f172a', pad=10)
            st.pyplot(fig)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Derived KPI breakdown
            st.markdown("""
                <div class="custom-card">
                    <div class="card-title">📊 Key Computed Metrics</div>
                    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
                        <div>
                            <div class="metric-label">Estimated Bill</div>
                            <div class="metric-value">₹{:.2f}</div>
                        </div>
                        <div>
                            <div class="metric-label">Service Stress</div>
                            <div class="metric-value">{:.3f}</div>
                        </div>
                        <div>
                            <div class="metric-label">Revenue Segment</div>
                            <div class="metric-value">{}</div>
                        </div>
                    </div>
                </div>
                """.format(total_charges, service_stress, revenue_segment), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
                <div class="custom-card" style="height:100%">
                    <div class="card-title">🎯 Retention Playbook & Action Plan</div>
                    <p style="color:#475569;font-size:0.95rem;margin-bottom:20px;">Tailored actions triggered by this customer's behavior profile:</p>
                """, unsafe_allow_html=True)
            
            # Action logic based on features
            actions = []
            if cust_service_calls >= 4:
                actions.append(("🚨 Proactive Support Call Escalation", "Customer has called support 4+ times. Trigger immediate callback from a senior retention specialist to resolve outstanding grievances."))
            elif cust_service_calls >= 2:
                actions.append(("🛠️ Proactive Service Healthcheck", "Customer service calls are mounting. Trigger a systems performance check on their lines and send a feedback survey with a priority voucher."))
                
            if intl_plan == "Yes" and intl_calls <= 2:
                actions.append(("✈️ International Plan Optimization", "The customer is paying for the International Plan but has very low usage (≤ 2 calls). Propose a cheaper standard package with pay-as-you-go international calls to improve billing satisfaction."))
                
            if total_charges >= metadata["revenue_bins"][2]: # High revenue segment
                actions.append(("👑 High-Value Customer VIP Status", "This customer falls into the High-Revenue segment. Enroll them in the VIP loyalty tier, providing free data add-ons and priority queuing."))
                
            if account_length >= 150:
                actions.append(("🎂 Milestone Loyalty Reward", "Account length is 150+ periods. Celebrate account maturity by offering a loyalty discount code or contract renewal discount."))
                
            if day_mins >= 250.0:
                actions.append(("⚡ Heavy Daytime Plan Offer", "High daytime minute usage detected. Pitch an unlimited voice calling add-on to ensure they don't face bill shock."))
                
            if not actions:
                actions.append(("💚 Standard Lifecycle Care", "Maintain customer satisfaction via routine checks, seasonal updates, and automated digital newsletter offers. No active intervention needed."))
                
            for title, desc in actions:
                st.markdown(f"""
                    <div class="rec-item">
                        <span class="rec-icon">⚡</span>
                        <div>
                            <strong style="color:#1e1b4b;font-size:1rem;">{title}</strong><br/>
                            <span style="color:#475569;font-size:0.9rem;">{desc}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
    # ------------------ TAB 2: MODEL INSIGHTS & EDA ------------------
    with tab2:
        st.subheader("Data-Driven Decision Support & Insights")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
                <div class="custom-card">
                    <div class="card-title">🌲 Decision Tree Feature Importance</div>
                """, unsafe_allow_html=True)
            
            # Feature Importance
            dt_clf = pipeline.named_steps["classifier"]
            feat_importances = pd.Series(dt_clf.feature_importances_, index=metadata["features"])
            top_features = feat_importances.sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=top_features.values, y=top_features.index, palette="viridis", ax=ax)
            ax.set_title("Top 10 Most Predictive Factors", fontsize=11, fontweight='bold', color='#0f172a')
            ax.set_xlabel("Gini Importance Score", fontsize=9, color='#475569')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("""
                <p style="color:#64748b;font-size:0.85rem;margin-top:10px;">
                <strong>Key Takeaway:</strong> The Decision Tree model heavily relies on <em>Total day minutes</em>, <em>Customer service calls</em>, and <em>International plan</em> status to predict churn. These represent the primary levers retention managers must monitor.
                </p>
                </div>
                """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
                <div class="custom-card">
                    <div class="card-title">📈 Churn Drivers Analysis</div>
                """, unsafe_allow_html=True)
            
            # Let's load df to generate some quick EDA summaries
            df_eda = pd.read_csv('dataset.csv')
            
            # Option to choose EDA chart
            eda_option = st.selectbox(
                "Select Feature for Churn Distribution Analysis",
                ["Customer service calls", "International plan", "Voice mail plan"]
            )
            
            fig, ax = plt.subplots(figsize=(6, 4.2))
            
            if eda_option == "Customer service calls":
                churn_by_calls = pd.crosstab(df_eda['Customer service calls'], df_eda['Churn'], normalize='index') * 100
                churn_by_calls.plot(kind='bar', stacked=True, color=['#10b981', '#ef4444'], ax=ax)
                ax.set_ylabel("Percentage (%)", fontsize=9)
                ax.set_title("Churn Rate by Number of Service Calls", fontsize=11, fontweight='bold', color='#0f172a')
                ax.legend(["Stayed", "Churned"])
            elif eda_option == "International plan":
                churn_by_intl = pd.crosstab(df_eda['International plan'], df_eda['Churn'], normalize='index') * 100
                churn_by_intl.plot(kind='bar', stacked=True, color=['#10b981', '#ef4444'], ax=ax)
                ax.set_ylabel("Percentage (%)", fontsize=9)
                ax.set_title("Churn Rate by International Plan Subscription", fontsize=11, fontweight='bold', color='#0f172a')
                ax.legend(["Stayed", "Churned"])
            else:
                churn_by_vmail = pd.crosstab(df_eda['Voice mail plan'], df_eda['Churn'], normalize='index') * 100
                churn_by_vmail.plot(kind='bar', stacked=True, color=['#10b981', '#ef4444'], ax=ax)
                ax.set_ylabel("Percentage (%)", fontsize=9)
                ax.set_title("Churn Rate by Voice Mail Plan Subscription", fontsize=11, fontweight='bold', color='#0f172a')
                ax.legend(["Stayed", "Churned"])
                
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
    # ------------------ TAB 3: BATCH PREDICTION PORT ------------------
    with tab3:
        st.subheader("High-Throughput Inference Port")
        st.markdown("""
            Upload customer call records to predict churn risks in bulk.
            The uploaded CSV should ideally match the columns of `dataset.csv`. Missing fields will be preprocessed automatically.
            """)
        
        # Download Template logic
        template_buffer = io.StringIO()
        df_eda = pd.read_csv('dataset.csv')
        df_eda.drop(columns=['Churn'], errors='ignore').head(5).to_csv(template_buffer, index=False)
        st.download_button(
            label="📥 Download Template CSV Structure",
            data=template_buffer.getvalue(),
            file_name="telecom_churn_input_template.csv",
            mime="text/csv"
        )
        
        st.write("---")
        
        uploaded_file = st.file_uploader("Drag and drop your customer CSV file here", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read uploaded CSV
                df_up = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                st.subheader("Data Preview")
                st.dataframe(df_up.head(10))
                
                if st.button("Run Batch Inference"):
                    with st.spinner("Processing customer profiles & scaling variables..."):
                        # Build batch inference dataframe using same logic as train_model.py
                        df_proc = df_up.copy()
                        
                        # Validate columns
                        required_cols = ['Account length', 'International plan', 'Voice mail plan', 
                                         'Total day minutes', 'Total day calls', 'Total day charge',
                                         'Total eve minutes', 'Total eve calls', 'Total eve charge',
                                         'Total night minutes', 'Total night calls', 'Total night charge',
                                         'Total intl minutes', 'Total intl calls', 'Total intl charge',
                                         'Customer service calls', 'State']
                        
                        missing_cols = [c for c in required_cols if c not in df_proc.columns]
                        if missing_cols:
                            st.warning(f"Note: Some columns were missing and will be filled with standard values: {missing_cols}")
                            # Impute standard/mean values for missing fields to ensure robust execution
                            for mc in missing_cols:
                                if mc in ['State']:
                                    df_proc[mc] = 'KS'
                                elif mc in ['International plan', 'Voice mail plan']:
                                    df_proc[mc] = 'No'
                                elif mc in ['Account length', 'Total day calls', 'Total eve calls', 'Total night calls']:
                                    df_proc[mc] = 100
                                elif mc in ['Total day minutes', 'Total eve minutes', 'Total night minutes']:
                                    df_proc[mc] = 180.0
                                elif mc in ['Total intl minutes']:
                                    df_proc[mc] = 10.0
                                elif mc in ['Total intl calls']:
                                    df_proc[mc] = 4
                                elif mc in ['Customer service calls']:
                                    df_proc[mc] = 1
                                else:
                                    # Charge fields
                                    df_proc[mc] = 0.0
                                    
                        # Derived columns
                        # Use charge values if available, else minutes * rate
                        if 'Total day charge' in df_proc.columns:
                            df_proc['Total Charges'] = df_proc['Total day charge'] + df_proc['Total eve charge'] + df_proc['Total night charge'] + df_proc['Total intl charge']
                        else:
                            df_proc['Total Charges'] = (df_proc['Total day minutes'] * 0.17) + (df_proc['Total eve minutes'] * 0.085) + (df_proc['Total night minutes'] * 0.045) + (df_proc['Total intl minutes'] * 0.27)
                            
                        df_proc['Total_Usage'] = df_proc['Total day minutes'] + df_proc['Total eve minutes'] + df_proc['Total night minutes'] + df_proc['Total intl minutes']
                        df_proc['Service_Stress'] = df_proc['Customer service calls'] / (df_proc['Account length'] + 1)
                        
                        # Bin segments using thresholds from metadata
                        bins = metadata["revenue_bins"]
                        df_proc['Revenue_Segment'] = df_proc['Total Charges'].apply(lambda x: get_revenue_segment(x, bins))
                        
                        # Map categorical binary
                        df_proc['International plan'] = df_proc['International plan'].map({'No': 0, 'Yes': 1, 0: 0, 1: 1}).fillna(0).astype(int)
                        df_proc['Voice mail plan'] = df_proc['Voice mail plan'].map({'No': 0, 'Yes': 1, 0: 0, 1: 1}).fillna(0).astype(int)
                        
                        # One-hot encode State and Revenue_Segment
                        df_proc_dummies = pd.get_dummies(df_proc, columns=['State', 'Revenue_Segment'], drop_first=True, dtype=int)
                        
                        # Reindex to match features
                        df_proc_dummies = df_proc_dummies.reindex(columns=metadata["features"], fill_value=0)
                        
                        # Predict
                        batch_preds = pipeline.predict(df_proc_dummies)
                        batch_probs = pipeline.predict_proba(df_proc_dummies)[:, 1]
                        
                        # Add results back to uploaded dataframe
                        df_results = df_up.copy()
                        df_results['Churn_Risk_Score'] = np.round(batch_probs * 100, 2)
                        df_results['Predicted_Churn'] = batch_preds
                        df_results['Predicted_Churn_Label'] = df_results['Predicted_Churn'].map({0: 'Loyal', 1: 'High Churn Risk'})
                        
                        st.subheader("Batch Prediction Results")
                        st.dataframe(df_results)
                        
                        # Export button
                        export_buffer = io.StringIO()
                        df_results.to_csv(export_buffer, index=False)
                        st.download_button(
                            label="📤 Download Prediction Results (CSV)",
                            data=export_buffer.getvalue(),
                            file_name="telecom_churn_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Visual statistics for the batch
                        st.write("---")
                        st.subheader("Batch Visual Summary")
                        col_char1, col_char2 = st.columns([1, 1])
                        
                        with col_char1:
                            # Churn distribution
                            churn_counts = df_results['Predicted_Churn_Label'].value_counts()
                            fig, ax = plt.subplots(figsize=(5, 4))
                            churn_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#10b981', '#ef4444'], ax=ax)
                            ax.set_ylabel("")
                            ax.set_title("Overall Churn Risk Distribution", fontsize=11, fontweight='bold')
                            st.pyplot(fig)
                            
                        with col_char2:
                            # Average Customer Service calls by churn label
                            fig, ax = plt.subplots(figsize=(5, 4))
                            sns.barplot(data=df_results, x='Predicted_Churn_Label', y='Customer service calls', palette=['#10b981', '#ef4444'], errorbar=None, ax=ax)
                            ax.set_xlabel("Customer Segment")
                            ax.set_ylabel("Avg Customer Service Calls")
                            ax.set_title("Customer Service Interaction vs Risk Profile", fontsize=11, fontweight='bold')
                            st.pyplot(fig)
                            
            except Exception as e:
                st.error(f"Error executing batch inference: {e}")
else:
    st.info("Awaiting training assets. Run train_model.py to initialize the predicting logic.")
