import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os  # Added to check file existence

# Get current date and timestamp
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

# Auto-refresh the Streamlit page every 2 seconds, up to 100 times
from streamlit_autorefresh import st_autorefresh
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

# FizzBuzz logic
if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

# Construct the file path
attendance_file = f"data/Attendance_{date}.csv"

# Check if the file exists
if os.path.exists(attendance_file):
    # Read and display the CSV file in a Streamlit DataFrame
    df = pd.read_csv(attendance_file)
    st.dataframe(df.style.highlight_max(axis=0))
else:
    st.write(f"File not found: {attendance_file}")
