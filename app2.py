# ---------- Import Required Libraries ----------
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ---------- Full CSS Injection ----------
st.markdown("""
    <style>
        /* Base Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main App Background */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            background-attachment: fixed !important;
            min-height: 100vh;
        }
        
        /* Remove Streamlit Default Padding */
        .main .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Main Content Container */
        .main-container {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 20px !important;
            padding: 30px !important;
            margin: 2rem auto !important;
            width: calc(100% - 40px) !important;
            max-width: 1400px !important;
            box-shadow: 0 12px 40px rgba(0,0,0,0.15) !important;
        }
        
        /* Header Styles */
        .app-header {
            text-align: center;
            padding: 2rem 0;
            color: white !important;
        }
        
        .app-header h1 {
            font-size: 3rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        }
        
        .app-header p {
            font-size: 1.2rem !important;
            opacity: 0.9 !important;
        }
        
        /* Sidebar Styles */
        .stSidebar {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin: 20px !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
            border: none !important;
        }
        
        /* Slider Styles */
        .stSlider {
            margin: 1.5rem 0 !important;
        }
        
        .stSlider label {
            display: block !important;
            margin-bottom: 0.5rem !important;
            font-weight: 500 !important;
            color: #555 !important;
            font-size: 1rem !important;
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid !important;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
            gap: 1.5rem !important;
            margin: 2rem 0 !important;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.98) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            text-align: center !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
            transition: transform 0.3s ease !important;
        }
        
        .stat-card:hover {
            transform: translateY(-5px) !important;
        }
        
        .stat-value {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #667eea !important;
            margin-bottom: 0.5rem !important;
        }
        
        .stat-label {
            color: #666 !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }
        
        /* Chart Cards */
        .chart-card {
            background: rgba(255, 255, 255, 0.98) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            margin-bottom: 2rem !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
        }
        
        /* Dataframe Styling */
        .stDataFrame {
            background: rgba(255, 255, 255, 0.98) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
            margin-bottom: 2rem !important;
        }
        
        /* Section Headers */
        .stSubheader {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            color: #333 !important;
            margin: 1.5rem 0 1rem 0 !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 2px solid rgba(102, 126, 234, 0.2) !important;
        }
        
        /* Fix for Streamlit's default styles */
        .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj {
            border: none !important;
        }
        
        div.stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
        }
        
        /* Custom table styles */
        .metric-table {
            background: white !important;
            border-radius: 10px !important;
            padding: 15px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        }
        
        .metric-table th {
            text-align: left !important;
            padding: 8px 12px !important;
            background: #f8f9fa !important;
            color: #555 !important;
            font-size: 13px !important;
        }
        
        .metric-table td {
            padding: 8px 12px !important;
            border-bottom: 1px solid #eee !important;
            font-size: 13px !important;
        }
        
        .metric-table tr:last-child td {
            border-bottom: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- App Header ----------
st.markdown("""
    <div class="app-header">
        <h1>🍔 Fast Food Association Mining</h1>
        <p>Discover patterns and relationships in fast food purchasing behavior</p>
    </div>
""", unsafe_allow_html=True)

# Main Content Container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("⚙️ Mining Parameters")
    min_support = st.slider("Minimum Support", 0.01, 0.2, 0.05, 0.01, 
                           help="Minimum support threshold for itemsets")
    min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05,
                             help="Minimum confidence threshold for rules")
    min_lift = st.slider("Minimum Lift", 1.0, 3.0, 1.2, 0.1,
                        help="Minimum lift threshold for rules")

# ---------- Data Loading ----------
@st.cache_data
def load_data():
    file_path = "fastfood_groceries_format.xlsx"
    df = pd.read_excel(file_path, header=None).fillna("")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    transactions = df.apply(lambda row: [item for item in row if item != ""], axis=1).tolist()
    return transactions

transactions = load_data()
te = TransactionEncoder()
df_encoded = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

# ---------- Apriori Algorithm ----------
@st.cache_data
def run_apriori(min_support, min_confidence, min_lift):
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True, max_len=3)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]
    return frequent_itemsets, rules

frequent_itemsets, rules = run_apriori(min_support, min_confidence, min_lift)

# ---------- Remove Redundant Rules ----------
def clean_rules(rules_df):
    seen = []
    indexes = []
    for i, row in rules_df.iterrows():
        combined = sorted(list(row['antecedents']) + list(row['consequents']))
        if combined not in seen:
            seen.append(combined)
            indexes.append(i)
    return rules_df.loc[indexes]

rules = clean_rules(rules)

# ---------- Item Filter ----------
selected_items = st.multiselect(
    "🔍 Filter combos by item(s)", 
    sorted(te.columns_),
    help="Filter rules by specific items in antecedents or consequents"
)

if selected_items:
    rules = rules[
        rules['antecedents'].apply(lambda x: any(item in x for item in selected_items)) |
        rules['consequents'].apply(lambda x: any(item in x for item in selected_items))
    ]

# ---------- Key Metrics ----------
st.markdown("""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{:,}</div>
            <div class="stat-label">Total Transactions</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">Unique Items</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">Frequent Itemsets</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">Association Rules</div>
        </div>
    </div>
""".format(len(transactions), len(te.columns_), len(frequent_itemsets), len(rules)), unsafe_allow_html=True)

# ---------- Top Itemsets Visualization ----------
if not frequent_itemsets.empty:
    st.subheader("📊 Top 10 Frequent Itemsets")
    
    # Prepare the data
    top_support = frequent_itemsets.sort_values(by="support", ascending=False).head(10)
    top_support['itemsets'] = top_support['itemsets'].apply(lambda x: ', '.join(list(x)))
    top_support['support_pct'] = (top_support['support'] * 100).round(2)
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Enhanced bar chart with custom styling
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create gradient effect for bars
        cmap = plt.get_cmap('BuPu')
        gradient = np.linspace(0, 1, len(top_support))
        colors = cmap(gradient)
        
        bars = ax.barh(
            top_support['itemsets'],
            top_support['support'],
            color=colors,
            edgecolor='white',
            linewidth=0.7,
            height=0.7
        )
        
        # Add value labels to bars
        for i, (value, name) in enumerate(zip(top_support['support'], top_support['itemsets'])):
            ax.text(
                value + 0.005, 
                i, 
                f'{value:.3f}',
                ha='left', 
                va='center',
                fontsize=10,
                color='#333'
            )
        
        # Custom styling
        ax.set_title('Most Frequent Item Combinations', 
                    fontsize=16, pad=20, fontweight='bold', color='#333')
        ax.set_xlabel('Support Value', fontsize=12, labelpad=10, color='#555')
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ddd')
        ax.spines['bottom'].set_color('#ddd')
        ax.tick_params(axis='y', colors='#555', labelsize=11)
        ax.tick_params(axis='x', colors='#555')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add some space at the top
        plt.subplots_adjust(top=0.9)
        
        st.pyplot(fig)
    
    with col2:
        # Enhanced table display with metrics
        st.markdown("""
            <div class="metric-table">
                <h4 style='margin-top:0; margin-bottom:15px; color:#333;'>Support Metrics</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Itemset</th>
                            <th>Support</th>
                        </tr>
                    </thead>
                    <tbody>
        """, unsafe_allow_html=True)
        
        for _, row in top_support.iterrows():
            st.markdown(f"""
                <tr>
                    <td>{row['itemsets'][:30]}{'...' if len(row['itemsets']) > 30 else ''}</td>
                    <td style='text-align:right;'>{row['support']:.3f}</td>
                </tr>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                    </tbody>
                </table>
            </div>
            
            <div style='margin-top:15px; padding:15px; background:#f8fafc; border-radius:10px;'>
                <p style='margin:0; font-size:13px; color:#555;'>
                    <b>Support</b> indicates how frequently the itemset appears in all transactions.
                </p>
            </div>
        """, unsafe_allow_html=True)

# ---------- Association Rules Display ----------
st.subheader("🔗 Top Association Rules")
if not rules.empty:
    display_rules = rules.copy()
    display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    st.dataframe(
        display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        .sort_values(by="lift", ascending=False)
        .head(10)
        .style.format({
            'support': '{:.3f}',
            'confidence': '{:.3f}',
            'lift': '{:.2f}'
        })
        .background_gradient(cmap='BuPu', subset=['support', 'confidence', 'lift']),
        height=400
    )
else:
    st.warning("No association rules found with current parameters. Try adjusting the thresholds.")

# ---------- Confidence vs Lift Scatter Plot ----------
if not rules.empty:
    st.subheader("📈 Confidence vs Lift Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=rules,
        x='confidence',
        y='lift',
        size='support',
        hue='support',
        palette='coolwarm',
        sizes=(20, 200),
        ax=ax
    )
    ax.set_title("Rule Strength Analysis", pad=20, fontweight='bold')
    ax.set_xlabel("Confidence", labelpad=10)
    ax.set_ylabel("Lift", labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Support')
    plt.tight_layout()
    st.pyplot(fig)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)
