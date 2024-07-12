import torch
import streamlit as st


def update_cuda_stats_at_progressbar():
    stats = torch.cuda.memory_stats()
    gpu_id = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    try:
        allocated_mem = stats["allocated_bytes.all.current"]
    except KeyError:
        return
    reserved_mem = stats["reserved_bytes.all.current"]
    free_mem = total_memory - reserved_mem
    vram = st.session_state.vram_empty
    vram.progress(
        (free_mem / total_memory),
        text=f"{free_mem / (1024 ** 3):.2f} GB, allocated: {allocated_mem / (1024 ** 3):.2f} GB, reserved: {reserved_mem / (1024 ** 3):.2f} GB"
    )
