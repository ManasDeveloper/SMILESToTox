import streamlit as st
from predict import predict
import matplotlib.pyplot as plt

# App title
st.set_page_config(
    page_title="SMILES2Tox - Toxicity Predictor",
    page_icon="🧪",
    layout="wide"
)

# --- MAIN TITLE ---
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🧪 SMILES2Tox: AI-powered Toxicity Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict chemical toxicity from SMILES strings using deep learning (BiLSTM)</p>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧬 Settings")
    threshold = st.slider(
        "Toxicity Threshold", 0.0, 1.0, 0.5, 0.01,
        help="Adjust the threshold above which the model classifies a target as toxic"
    )
    st.markdown("You can adjust this if you want a stricter or more lenient toxicity cutoff.")
    st.markdown("Made by Manas Kulkarni")

# --- SMILES Input ---
st.subheader("🔍 Enter a SMILES String")
smiles_input = st.text_input("Example: `C1=CC=CC=C1`", placeholder="Paste your SMILES string here...")

if st.button("🧠 Predict Toxicity"):
    if not smiles_input.strip():
        st.warning("Please enter a valid SMILES string.")
    else:
        with st.spinner("Running model inference..."):
            results = predict(smiles_input.strip(), threshold=threshold)

            st.success("Prediction complete!")

            st.subheader("📋 Toxicity Prediction per Target")

            col1, col2 = st.columns(2)
            with col1:
                for res in results[:6]:
                    label = "🟥 Toxic" if res["is_toxic"] else "🟩 Non-toxic"
                    st.markdown(f"**{res['target']}**: {res['score']:.3f} {label}")

            with col2:
                for res in results[6:]:
                    label = "🟥 Toxic" if res["is_toxic"] else "🟩 Non-toxic"
                    st.markdown(f"**{res['target']}**: {res['score']:.3f} {label}")

            # Plotting
            st.subheader("📊 Toxicity Probability Chart")
            targets = [r["target"] for r in results]
            scores = [r["score"] for r in results]
            colors = ['#e74c3c' if r["is_toxic"] else '#2ecc71' for r in results]  # red for toxic, green for non-toxic

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(targets, scores, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Predicted Toxicity Score")
            ax.invert_yaxis()
            for i, bar in enumerate(bars):
                ax.text(bar.get_width() + 0.01, bar.get_y() + 0.3, f"{scores[i]:.2f}", fontsize=9)
            st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 0.9em;'>© 2025 SMILES2Tox • Built with 🧠 PyTorch & Streamlit</p>",
    unsafe_allow_html=True
)
