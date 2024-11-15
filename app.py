import streamlit as st
import pandas as pd
import time 
from datetime import datetime

import win32com.client
from win32com.client import Dispatch
"""streamlit: This library is used to create web apps with Python. You’ll use Streamlit to display things like text and dataframes on a webpage.
pandas: A powerful data manipulation library used to read and display data (in this case, from a CSV file).
time: Helps with time-related functions, like getting the current time.
datetime: Used to format time and dates, for instance to get a human-readable date and timestamp.
win32com.client: This allows interaction with Windows COM objects, like using the text-to-speech feature in Windows (SAPI.SpVoice)."""
ts=time.time()
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
"""ts = time.time(): Gets the current time as a timestamp (a large number representing seconds since 1970).
date: Converts the timestamp into a human-readable date (format: "day-month-year").
timestamp: Converts the timestamp into a time string (format: "hours:minutes
")."""

from streamlit_autorefresh import st_autorefresh

count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")
"""st_autorefresh: A function from the streamlit_autorefresh library that refreshes the web page at a given interval.
interval=2000: Sets the refresh rate to 2000 milliseconds (2 seconds).
limit=100: Limits the refresh counter to 100 times.
key="fizzbuzzcounter": This is used to keep track of the refreshes with the name "fizzbuzzcounter".
count: Stores the number of times the page has refreshed.
"""
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

"""This block implements the "FizzBuzz" game.
If the counter (count) is:
0: Displays "Count is zero".
A multiple of both 3 and 5: Displays "FizzBuzz".
A multiple of 3 but not 5: Displays "Fizz".
A multiple of 5 but not 3: Displays "Buzz".
Otherwise, it just displays the current count."""

df=pd.read_csv("face_detect\Attendance_21-10-2024.csv")


"""pd.read_csv(): Reads a CSV file (a text file with rows of data) into a DataFrame.
"data\Attendance_21-10-2024.csv" + date + ".csv": This creates the file path for a CSV file by combining the base name "Attendance_21-10-2024" with today’s date (date) and .csv."""

st.dataframe(df.style.highlight_max(axis=0))

"""st.dataframe(): Displays the DataFrame (df) as a table in the Streamlit web app.
df.style.highlight_max(axis=0): Highlights the maximum value in each column of the DataFrame."""

"""This code creates a simple web page that auto-refreshes every 2 seconds up to 100 times.
It implements a "FizzBuzz" game based on the refresh count.
It reads an attendance CSV file (specific to the current date) and displays it in a table, highlighting the highest values in each column."""