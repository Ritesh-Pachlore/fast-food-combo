# ---------- Import Required Libraries ----------
import streamlit as st  # For creating the interactive web app
import pandas as pd  # For data manipulation
import seaborn as sns  # For creating visualizations
import matplotlib.pyplot as plt  # For plotting graphs
from mlxtend.frequent_patterns import apriori, association_rules  # For mining frequent itemsets and generating rules
from mlxtend.preprocessing import TransactionEncoder  # For transforming data into a format suitable for Apriori

# ---------- Inject Custom HTML/CSS Styling for the Streamlit App ----------
st.markdown("""
    <style>
        body {
            background-color: #000000;  /* Dark background */
            color: #FFFFFF;  /* Light text color */
            font-family: 'Inter', sans-serif;
        }
        .main {
            background-color: rgba(30, 30, 30, 0.95);  /* Semi-transparent panel */
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 40px rgba(0,0,0,0.3);  /* Soft shadow for depth */
            margin-top: 20px;
        }
        h1 {
            color: #FFFFFF;
            text-align: center;
            font-size: 3rem;
            text-shadow: 2px 2px 4px rgba(255,255,255,0.2);  /* Text shadow effect */
        }
        .stSlider > label, .stMultiSelect > label {
            font-weight: bold;
            color: #FFFFFF;
        }
        .css-1d391kg, .css-1kyxreq {
            background-color: #111111 !important;
            color: #FFFFFF !important;
        }
        .stMetric label {
            color: #FFFFFF !important;
        }
        .stDataFrame, .stDataFrame table {
            color: #FFFFFF !important;
            background-color: #222222 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Title of the Web App ----------
st.title("🍔 Fast Food Association Minning")

# ---------- Sidebar for Adjustable Parameters ----------
with st.sidebar:
    st.header("⚙️ Parameters")
    min_support = st.slider("Minimum Support", 0.01, 0.2, 0.05, 0.01)  # Minimum support threshold for Apriori
    min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)  # Minimum confidence threshold
    min_lift = st.slider("Minimum Lift", 1.0, 3.0, 1.2, 0.1)  # Minimum lift threshold

# ---------- Load Transaction Dataset from Excel File ----------
file_path = "fastfood_groceries_format.xlsx"  # Path to dataset
df = pd.read_excel(file_path, header=None).fillna("")  # Read data and fill empty cells with empty strings

# ---------- Transform Dataset to Transaction Format ----------
transactions = df.apply(lambda row: [item for item in row if item != ""], axis=1).tolist()  # Convert each row to a list of items
te = TransactionEncoder()  # Create a TransactionEncoder instance
df_encoded = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)  # One-hot encode the transactions

# ---------- Run the Apriori Algorithm ----------
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True, max_len=3)  # Find frequent itemsets up to size 3
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)  # Generate rules based on confidence
rules = rules[rules['lift'] >= min_lift]  # Filter rules by lift threshold

# ---------- Remove Redundant or Duplicate Rules ----------
def clean_rules(rules_df):
    seen = []
    indexes = []
    for i, row in rules_df.iterrows():
        combined = sorted(list(row['antecedents']) + list(row['consequents']))  # Combine antecedents and consequents
        if combined not in seen:  # If not already seen, keep the rule
            seen.append(combined)
            indexes.append(i)
    return rules_df.loc[indexes]

rules = clean_rules(rules)

# ---------- Filter Rules by User-Selected Items ----------
selected_items = st.multiselect("🔍 Filter combos by item(s)", sorted(te.columns_))
if selected_items:
    rules = rules[
        rules['antecedents'].apply(lambda x: any(item in x for item in selected_items)) |
        rules['consequents'].apply(lambda x: any(item in x for item in selected_items))
    ]

# ---------- Dashboard Summary Metrics ----------
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", len(transactions))  # Number of total transactions
    col2.metric("Unique Items", len(te.columns_))  # Number of unique items across all transactions
    col3.metric("Frequent Itemsets", len(frequent_itemsets))  # Number of frequent itemsets found
    col4.metric("Association Rules", len(rules))  # Number of association rules generated

# ---------- Open Main Display Container ----------
st.markdown('<div class="main">', unsafe_allow_html=True)

# ---------- Bar Chart: Top 10 Frequent Itemsets ----------
if not frequent_itemsets.empty:
    st.subheader("📊 Top 10 Frequent Itemsets")
    top_support = frequent_itemsets.sort_values(by="support", ascending=False).head(10)  # Top 10 by support
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.barplot(
        y=top_support['itemsets'].apply(lambda x: ', '.join(list(x))),  # Convert frozenset to string
        x=top_support['support'], palette='viridis', ax=ax1)
    ax1.set_title("Top Frequent Itemsets by Support")
    ax1.set_xlabel("Support")
    st.pyplot(fig1)  # Display plot in Streamlit

# ---------- Table: Top Association Rules ----------
st.subheader("🔗 Top Association Rules")
if not rules.empty:
    display_rules = rules.copy()
    # Convert frozensets to comma-separated strings for display
    display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    # Show top 10 rules sorted by lift
    st.dataframe(display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by="lift", ascending=False).head(10))
else:
    st.info("No rules found for current filters.")  # Show message if no rules meet criteria

# ---------- Scatter Plot: Confidence vs Lift ----------
if not rules.empty:
    st.subheader("📈 Confidence vs Lift")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=rules, x="confidence", y="lift", hue="support", palette="cool", ax=ax2)
    ax2.set_title("Rule Strength Scatter Plot")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Lift")
    st.pyplot(fig2)  # Display plot

# ---------- Close Custom Styled Container ----------
st.markdown('</div>', unsafe_allow_html=True)
