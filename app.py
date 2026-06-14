import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# --- Model Architecture (Copy these classes directly from your notebook) ---
# It's crucial that these definitions are identical to the ones used for training.

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output

class CIN(nn.Module):
    def __init__(self, field_dim, cin_layer_units):
        super(CIN, self).__init__()
        self.field_dim = field_dim
        self.cin_layer_units = cin_layer_units
        self.conv_layers = nn.ModuleList()
        prev_dim = field_dim
        for unit in cin_layer_units:
            self.conv_layers.append(nn.Conv1d(prev_dim * field_dim, unit, kernel_size=1))
            prev_dim = unit

    def forward(self, inputs):
        batch_size, field_dim, embed_dim = inputs.shape
        hidden_layers = [inputs]
        final_layers = []
        prev_dim_Xi = field_dim
        for i, conv in enumerate(self.conv_layers):
            X0 = hidden_layers[0]
            Xi = hidden_layers[-1]
            interaction = X0.unsqueeze(2) * Xi.unsqueeze(1)
            cin_input = interaction.view(batch_size, field_dim * prev_dim_Xi, embed_dim)
            xl = conv(cin_input)
            xl = F.relu(xl)
            xl_pooled = torch.sum(xl, dim=2)
            final_layers.append(xl_pooled)
            prev_dim_Xi = self.cin_layer_units[i]
            if i < len(self.conv_layers) - 1:
                hidden_layers.append(xl)
        final_output = torch.cat(final_layers, dim=1)
        return final_output

class CA_MCGRS_MHA_xDeepFM(nn.Module):
    def __init__(self, num_users, num_items, num_contexts, embed_dim=64,
                 num_heads=4, dnn_hidden_units=[256, 128], cin_layer_units=[128, 64]):
        super(CA_MCGRS_MHA_xDeepFM, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cin_layer_units = cin_layer_units
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.context_embeddings = nn.ModuleList([
            nn.Embedding(size, embed_dim) for size in num_contexts
        ])
        self.criteria_linear = nn.Linear(3, embed_dim)
        total_features = 4
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.cin = CIN(total_features, cin_layer_units)
        dnn_input_dim = total_features * embed_dim
        self.dnn_layers = nn.ModuleList()
        prev_dim = dnn_input_dim
        for hidden_dim in dnn_hidden_units:
            self.dnn_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.dnn_layers.append(nn.ReLU())
            self.dnn_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        self.linear = nn.Linear(total_features * embed_dim, 64)
        cin_output_dim = sum(self.cin_layer_units)
        dnn_output_dim = dnn_hidden_units[-1] if dnn_hidden_units else dnn_input_dim
        linear_output_dim = 64
        self.final_linear = nn.Linear(cin_output_dim + dnn_output_dim + linear_output_dim, 1)

    def forward(self, user_ids, item_ids, contexts, criteria):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        context_embeds_list = []
        for i, context_embed_layer in enumerate(self.context_embeddings):
            context_embeds_list.append(context_embed_layer(contexts[:, i]))
        context_embeds = torch.stack(context_embeds_list, dim=1).mean(dim=1)
        criteria_embeds = self.criteria_linear(criteria)
        all_embeds = torch.stack([
            user_embeds, item_embeds, context_embeds, criteria_embeds
        ], dim=1)
        attn_output = self.attention(all_embeds, all_embeds, all_embeds)
        attn_output = self.layer_norm(attn_output + all_embeds)
        cin_output = self.cin(attn_output)
        dnn_input = attn_output.view(attn_output.size(0), -1)
        dnn_output = dnn_input
        for layer in self.dnn_layers:
            dnn_output = layer(dnn_output)
        linear_output = F.relu(self.linear(dnn_input))
        combined_output = torch.cat([cin_output, dnn_output, linear_output], dim=1)
        prediction = torch.sigmoid(self.final_linear(combined_output)) * 4 + 1
        return prediction.squeeze(-1)

# --- Streamlit Caching ---
# Cache data loading to speed up app
@st.cache_data
def load_data():
    ratings_path = os.path.join('dataset', 'ratings.csv')
    items_path = os.path.join('dataset', 'items.csv')
    
    ratings_df = pd.read_csv(ratings_path)
    items_df = pd.read_csv(items_path)
    
    # This line is no longer needed after fixing the column names
    # st.write("Columns found in items.csv:", items_df.columns.tolist())
    
    # --- THIS IS THE CORRECTED LINE ---
    # It now uses 'Title' for the name and 'Item' for the ID, matching your file.
    item_id_to_name = pd.Series(items_df['Title'].values, index=items_df['Item']).to_dict()
    
    return ratings_df, item_id_to_name
# Cache model and encoder loading
@st.cache_resource
def load_model_and_encoders(ratings_df):
    # Re-create encoders to ensure they are fitted on the same data as during training
    encoders = {}
    encoders['user'] = LabelEncoder().fit(ratings_df['UserID'])
    encoders['item'] = LabelEncoder().fit(ratings_df['Item'])
    encoders['course'] = LabelEncoder().fit(ratings_df['Class'])
    encoders['semester'] = LabelEncoder().fit(ratings_df['Semester'])
    encoders['lockdown'] = LabelEncoder().fit(ratings_df['Lockdown'])

    # Get model parameters
    num_users = len(encoders['user'].classes_)
    num_items = len(encoders['item'].classes_)
    context_dims = [
        len(encoders['course'].classes_),
        len(encoders['semester'].classes_),
        len(encoders['lockdown'].classes_)
    ]

    # Instantiate the model architecture
    model = CA_MCGRS_MHA_xDeepFM(
        num_users=num_users,
        num_items=num_items,
        num_contexts=context_dims,
        embed_dim=64,
        num_heads=4,
        dnn_hidden_units=[256, 128, 64],
        cin_layer_units=[128, 64]
    )
    # Load the trained weights
    model.load_state_dict(torch.load('mcgrs_model.pth', map_location=torch.device('cpu')))
    model.eval() # Set model to evaluation mode
    return model, encoders

# --- Helper Functions ---
def get_star_rating(score, max_stars=5):
    """Converts a score to a star rating string."""
    rounded_score = np.round(score * 2) / 2
    full_stars = int(rounded_score)
    half_star = "★" if rounded_score - full_stars >= 0.5 else ""
    empty_stars = "☆" * (max_stars - full_stars - len(half_star))
    return f"{'★' * full_stars}{half_star}{empty_stars}"


# --- Main App ---
st.set_page_config(layout="wide", page_title="ITM-Rec Recommender")

# Load all necessary data and models
ratings, item_id_to_name_map = load_data()
model_instance, data_encoders = load_model_and_encoders(ratings)

# --- Sidebar for User Inputs ---
st.sidebar.header("👤 Select Your Profile & Context")

# Get unique sorted lists for dropdowns
user_list = sorted(ratings['UserID'].unique())
course_list = sorted(ratings['Class'].unique())
semester_list = sorted(ratings['Semester'].unique())
lockdown_list = sorted(ratings['Lockdown'].unique())
all_item_ids = ratings['Item'].unique()

selected_user_id = st.sidebar.selectbox("Select User ID", user_list)
selected_course = st.sidebar.selectbox("Select Course", course_list)
selected_semester = st.sidebar.selectbox("Select Semester", semester_list)
selected_lockdown = st.sidebar.selectbox("Select Lockdown Status", lockdown_list)
top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 3)

# --- Main Page ---
st.title("🛠️ ITM-Rec: Context-Aware Tool Recommender")
st.markdown("This web app demonstrates the **CA-MCGRS MHA-xDeepFM** model. Select a user and a context in the sidebar to get personalized tool recommendations.")
st.markdown("---")


if st.sidebar.button("Get Recommendations", type="primary"):
    # Find items the user has already rated
    rated_items = ratings[ratings['UserID'] == selected_user_id]['Item'].unique()
    # Find items to predict (all items minus the ones already rated)
    items_to_predict = np.setdiff1d(all_item_ids, rated_items)

    # Encode the inputs
    user_encoded = data_encoders['user'].transform([selected_user_id])[0]
    course_encoded = data_encoders['course'].transform([selected_course])[0]
    semester_encoded = data_encoders['semester'].transform([selected_semester])[0]
    lockdown_encoded = data_encoders['lockdown'].transform([selected_lockdown])[0]
    items_encoded = data_encoders['item'].transform(items_to_predict)
    
    # Prepare tensors for the model
    user_tensor = torch.tensor([user_encoded] * len(items_encoded), dtype=torch.long)
    item_tensor = torch.tensor(items_encoded, dtype=torch.long)
    context_tensor = torch.tensor([[course_encoded, semester_encoded, lockdown_encoded]] * len(items_encoded), dtype=torch.long)
    
    # For prediction, we don't have user's criteria ratings. We use a neutral average (e.g., 3.0) as input.
    criteria_tensor = torch.tensor([[3.0, 3.0, 3.0]] * len(items_encoded), dtype=torch.float32)

    with torch.no_grad():
        predictions = model_instance(user_tensor, item_tensor, context_tensor, criteria_tensor)
        
    # Create a DataFrame with results
    results_df = pd.DataFrame({
        'Item': items_to_predict,
        'PredictedRating': predictions.numpy()
    })
    
    top_recommendations = results_df.sort_values(by='PredictedRating', ascending=False).head(top_n)
    
    st.subheader(f"🌟 Top {top_n} Recommendations for User `{selected_user_id}`")
    st.info(f"**Context:** Course: `{selected_course}` | Semester: `{selected_semester}` | Lockdown: `{selected_lockdown}`")
    
    # Display recommendations in columns
    cols = st.columns(top_n)
    for i, (idx, row) in enumerate(top_recommendations.iterrows()):
        with cols[i]:
            item_name = item_id_to_name_map.get(row['Item'], f"Item ID {row['Item']}")
            rating_score = row['PredictedRating']
            star_display = get_star_rating(rating_score)
            
            st.markdown(f"#### {i+1}. {item_name}")
            st.metric(label="Predicted Rating", value=f"{rating_score:.2f} / 5")
            st.markdown(f"**{star_display}**")
            
    # --- Details Expander ---
    with st.expander("🔬 See Model Details & Inputs"):
        st.write("**Model Architecture:**")
        st.code(str(model_instance))
        
        st.write("**Sample Input Tensors (for the first predicted item):**")
        first_item_id = top_recommendations['Item'].iloc[0]
        first_item_encoded = data_encoders['item'].transform([first_item_id])[0]
        
        st.json({
            "Selected User ID": selected_user_id,
            "Encoded User ID": user_encoded,
            "Predicted Item ID": int(first_item_id),
            "Encoded Item ID": int(first_item_encoded),
            "Context (Course, Semester, Lockdown)": f"{selected_course}, {selected_semester}, {selected_lockdown}",
            "Encoded Context Tensor": [course_encoded, semester_encoded, lockdown_encoded],
            "Assumed Criteria Tensor": [3.0, 3.0, 3.0]
        })