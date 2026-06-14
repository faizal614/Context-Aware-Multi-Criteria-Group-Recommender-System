import streamlit as st
import torch
import torch.nn as nn
import pandas as pd

# ============================
# Model definition (adjust if your notebook has a different class)
# ============================
class MCGRSModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(MCGRSModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_id, item_id):
        user_vec = self.user_embedding(user_id)
        item_vec = self.item_embedding(item_id)
        x = torch.cat([user_vec, item_vec], dim=-1)
        return self.fc(x)

# ============================
# Load dataset info
# ============================
ratings_df = pd.read_csv("dataset/ratings.csv")   # <-- adjust path if needed

num_users = ratings_df['UserID'].nunique()
num_items = ratings_df['Item'].nunique()

# Map UserID and Item to indices (important for embeddings)
user_mapping = {u: i for i, u in enumerate(ratings_df['UserID'].unique())}
item_mapping = {i: j for j, i in enumerate(ratings_df['Item'].unique())}

# ============================
# Load trained model
# ============================
model = MCGRSModel(num_users, num_items)
model.load_state_dict(torch.load("mcgrs_model.pth", map_location="cpu"))
model.eval()

# ============================
# Streamlit UI
# ============================
st.title("🎯 Multi-Criteria Group Recommendation System (MCGRS)")

user_id_input = st.number_input("Enter User ID", 
                                min_value=int(ratings_df['UserID'].min()), 
                                max_value=int(ratings_df['UserID'].max()))

top_n = st.slider("How many recommendations?", 1, 10, 5)

if st.button("Get Recommendations"):
    if user_id_input not in user_mapping:
        st.error("User ID not found in dataset!")
    else:
        user_idx = user_mapping[user_id_input]
        item_ids = list(item_mapping.values())
        user_ids = [user_idx] * len(item_ids)

        user_tensor = torch.tensor(user_ids)
        item_tensor = torch.tensor(item_ids)

        with torch.no_grad():
            preds = model(user_tensor, item_tensor).squeeze()

        preds = preds.numpy()
        top_items_idx = preds.argsort()[-top_n:][::-1]

        # Map back to original Item IDs
        reverse_item_mapping = {v: k for k, v in item_mapping.items()}
        recommended_items = [reverse_item_mapping[idx] for idx in top_items_idx]

        st.write(f"### ✅ Recommended Items for User {user_id_input}")
        st.write(recommended_items)

