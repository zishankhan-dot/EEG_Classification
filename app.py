import streamlit as st
import matplotlib.pyplot as plt
from src.preprocessing import load_raw, load_demo, get_filteredBoth
from src.epochs import get_epochs, get_psd_condition
from src.plots import plot_raw, plotRawFilter, plotMotorSensors, plotConfusionMatrix, plotPredictionTimeline
from src.predict import run_pipeline
SENSORS = ["C3", "C4", "Cz"]

st.markdown("""
    <style>
        * { border-radius: 0px !important; }
        .block-container {
            padding-top: 1.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)
st.title("EEG Motor Imagery Predictor")
st.text("PhysioNet Dataset to predict Left/Right Motor Imagery on [C3,C4,Cz] channels")

# Sidebar
with st.sidebar:
    st.title("Controls")
    st.divider()
    st.subheader("Data Source")
    uploaded = st.file_uploader("EEG file (.edf)", type=["edf"], label_visibility="collapsed")
    st.caption("PhysioNet runs 4, 8, or 12 only (left/right hand imagery)")
    demo_btn = st.button("Load Demo Subject (S029 R04)", use_container_width=True)

if demo_btn:
    st.session_state['use_demo'] = True
if uploaded is not None:
    st.session_state['use_demo'] = False

use_demo = st.session_state.get('use_demo', False)

# Resolve raw
if uploaded is not None:
    raw = load_raw(uploaded)
elif use_demo:
    with st.spinner("Downloading demo file (S029 R04)..."):
        raw = load_demo()
    st.success("Demo loaded: Subject 29, Run 4 — left/right hand imagery")
else:
    st.warning(
        "Upload a PhysioNet EEG file (.edf) from the sidebar, "
        "or click **Load Demo Subject** to explore with a sample recording (Subject 29, Run 4)."
    )
    st.stop()

# Pipeline
raw_filtered     = get_filteredBoth(raw)
epochs, eventid  = get_epochs(raw_filtered, SENSORS)
psd_cond         = get_psd_condition(epochs, eventid)

annotations = list(set(a['description'] for a in raw.annotations))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sampling Frequency", f"{int(raw.info['sfreq'])} Hz")
c2.metric("Total Channels", len(raw.ch_names))
c3.metric("Recording Duration", f"{int(raw.times[-1])} s")
c4.metric("Annotations", ", ".join(sorted(annotations)))

with st.sidebar:
    st.divider()
    st.subheader("Visualisation")
    max_time = int(raw.times[-1]) - 10
    option   = st.selectbox("Channel", SENSORS)
    t_start  = st.slider("Window start (s)", 0, max_time, 0)
    t_end    = t_start + 10
    st.caption(f"Showing {t_start} s → {t_end} s")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Raw Signal")
    fig = plot_raw(raw, t_start, t_end, option)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with col2:
    st.subheader("2. Raw vs Filtered")
    fig = plotRawFilter(raw, t_start, t_end, option)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with col3:
    st.subheader("3. Motor Cortex")
    fig = plotMotorSensors(raw)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

st.subheader("Pipeline")
st.graphviz_chart("""
    digraph {
        rankdir=LR
        node [shape=box style=filled fillcolor="#1e1e2e" fontcolor=white color="#444444" fontname=TimesNewRoman]
        edge [color="#555555"]

        A [label="Upload\n.edf | Runs 4,8,12"]
        B [label="Filter\n8-13 Hz | 13-30 Hz"]
        C [label="Epochs\n 4s | T1, T2 → X,Y"]
        D [label="Features\n Welch PSD | Band Power"]
        E [label="Model\n Pre-trained LDA/ SVM / RF / MLP"]
        F [label="Predict\n Epoch table | T1/T2"]
        G [label="Metrics\n Acc | F1 | Confusion"]

        A -> B -> C -> D -> E -> F -> G
    }
""")

st.divider()
st.subheader("Model Inference")

with st.sidebar:
    st.divider()
    st.subheader("Model")
    model_name = st.selectbox("Select Model", ["LDA", "SVM", "RF", "MLP"])
    run_btn    = st.button("Run Prediction", use_container_width=True)

if run_btn:
    with st.spinner("Running pipeline..."):
        table, acc, f1, cm = run_pipeline(raw, model_name)

    if table is None:
        st.error("No T1/T2 epochs found in this file. Make sure you uploaded runs 4, 8, or 12.")
    else:
        correct = (table['Correct'] == 'Yes').sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",       f"{acc:.1%}")
        m2.metric("F1 Score",       f"{f1:.3f}")
        m3.metric("Correct Epochs", f"{correct} / {len(table)}")

        st.divider()
        res_col, cm_col = st.columns([2, 1])

        with res_col:
            st.markdown("**Epoch Predictions**")
            st.dataframe(
                table.style.apply(
                    lambda col: ['color: #4CAF50' if v == 'Yes' else 'color: #f44336' for v in col]
                    if col.name == 'Correct' else ['' for _ in col], axis=0
                ),
                use_container_width=True, hide_index=True
            )

        with cm_col:
            st.markdown("**Confusion Matrix**")
            fig = plotConfusionMatrix(cm)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("**Prediction Timeline**")
        fig = plotPredictionTimeline(table)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
