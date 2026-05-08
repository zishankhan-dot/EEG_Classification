import streamlit as st
import matplotlib.pyplot as plt
from src.preprocessing import load_raw, get_filteredBoth
from src.epochs import get_epochs, get_psd_condition
from src.plots import plot_psd, plot_psd_by_channel, plot_raw, plot_raw_vs_filtered

SENSORS = ["C3", "C4", "Cz"]

st.title("EEG Visualizer")

# Sidebar
with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload EEG file (.edf)", type=["edf"])

# Main area
if uploaded is not None:
    raw          = load_raw(uploaded)
    raw_filtered = get_filteredBoth(raw)
    epochs, eventid = get_epochs(raw_filtered, SENSORS)
    psd_cond     = get_psd_condition(epochs, eventid)

    with st.sidebar:
        max_time = int(raw.times[-1]) - 7
        t_start  = st.slider("Window start (s)", 0, max_time, 40)
        t_end    = t_start + 7
        st.info(f"Showing {t_start}s → {t_end}s")

    st.subheader("1. Raw Signal")
    fig = plot_raw(raw, t_start, t_end)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("2.Filtered")
    fig = plot_raw_vs_filtered(raw_filtered, SENSORS, t_start, t_end)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("3. PSD — All Channels")
    fig = plot_psd(raw, SENSORS)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("4. PSD by Channel")
    fig = plot_psd_by_channel(psd_cond, SENSORS)
    st.pyplot(fig)
    plt.close(fig)

