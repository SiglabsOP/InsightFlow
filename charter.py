import matplotlib
matplotlib.use('TkAgg')  # Use QtAgg for better event handling
 
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib.widgets import CheckButtons, Button
import mplcursors
import matplotlib.ticker as ticker
from matplotlib.widgets import Cursor
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches 
import pygame
 
     
import tkinter as tk
from tkinter import ttk
import threading
import os
import time
import joblib  # Install with `pip install joblib`
import asyncio
import shutil
import sys
import subprocess  # Make sure subprocess is imported
# Declare all global variables for plot elements
pygame.mixer.init()
# Play the custom .wav sound on startup
pygame.mixer.music.load("startup.wav")  # Replace with your .wav file path
pygame.mixer.music.play()
 
 
    
 
    
    
 
    
    
fig, ax1 = None, None
line_close, line_short_sma, line_long_sma, line_ema = None, None, None, None
line_upper_bb, line_lower_bb, line_chikou = None, None, None
scatter_sar, cloud_bullish, cloud_bearish = None, None, None
from matplotlib.widgets import Slider
from tkinter import scrolledtext
USE_THREADING = True  # Set to True to enable threading
hover_last_update = 0  # Initialize at the start of the script
tickerr = 'SPY'  # Default value, dynamically updated by target.py
def validate_and_clean_cache():
    """
    Validate the cache directory and remove corrupted or incompatible files.
    Automatically restarts the program if cache corruption is detected.
    """
    cache_dir = 'cache'
    corrupted = False
    expected_columns = {'Adj Close_SPY', 'Close_SPY', 'High_SPY', 'Low_SPY', 'Open_SPY', 'Volume_SPY'}
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, file)
            if file.endswith('.pkl'):  # Only check .pkl files
                try:
                    print(f"Validating cache file: {file}")
                    data = joblib.load(file_path)  # Attempt to load the file
                    # Check for required columns
                    if not expected_columns.issubset(set(data.columns)):
                        print(f"Incompatible cache detected: {file}. Missing columns.")
                        corrupted = True
                        break
                except Exception as e:
                    print(f"Corrupted cache detected: {file}. Error: {e}")
                    corrupted = True
                    break
        if corrupted:
            print("Cache corruption detected. Deleting all cache files...")
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                os.remove(file_path)
            print("Cache cleaned successfully. Restarting the program...")
            restart_program()  # Restart the script
    else:
        print("No cache directory found. Proceeding as normal.")
def restart_program():
    """
    Restart the current program.
    """
    try:
        # Get the full path of the Python executable
        python_executable = sys.executable
        # Ensure to pass the correct Python executable and script
        subprocess.Popen([python_executable] + sys.argv)
        sys.exit()
    except Exception as e:
        print(f"Failed to restart the program: {e}")
 
def launch_target(event=None):
    # Replace 'python' with 'python3' if required by your system
    os.system("python target.py &")
 
fig = None  # Declare fig globally
NOTES_FILE = "notes.txt"  # File to store the notes
 
# Update global plot settings
plt.rcParams.update({
    'font.size': 10,           # Base font size for all text
    'axes.titlesize': 12,      # Font size for subplot titles
    'axes.labelsize': 10,      # Font size for x and y axis labels
    'xtick.labelsize': 8,      # Font size for x-axis tick labels
    'ytick.labelsize': 8,      # Font size for y-axis tick labels
    'legend.fontsize': 9       # Font size for legends
}) 
 
def close_program(window):
    window.destroy()
from concurrent.futures import ThreadPoolExecutor
# Global ThreadPoolExecutor for managing threads
executor = ThreadPoolExecutor(max_workers=15)
 
 
 
def fetch_data_threaded(ticker, period, interval, callback=None):
    """
    Fetch data using threading or synchronous execution based on USE_THREADING.
    If threading is enabled, it uses ThreadPoolExecutor to run tasks in the background.
    Parameters:
        ticker (str): Stock ticker symbol (e.g., "SPY").
        period (str): Data period (e.g., "1y").
        interval (str): Data interval (e.g., "1d").
        callback (function): Optional callback to process the fetched data.
    """
    def task():
        # Use the existing get_data function to fetch actual data
        print(f"Fetching data for {ticker} with period={period} and interval={interval}...")
        data = get_data(ticker, period, interval)
        print("Data fetching complete.")
        print("Fetched data:")
        print(data.head())
        print(data.columns)
        return data  # Return the actual data
    def on_complete(future):
        # Callback to process the result
        try:
            data = future.result()  # Retrieve the fetched data
            print(f"Data fetched successfully for {ticker}.")
            if callback:
                callback(data)  # Pass the data to the callback
        except Exception as e:
            print(f"Error during data fetching: {e}")
    if USE_THREADING:
        # Use ThreadPoolExecutor to run the task
        future = executor.submit(task)
        future.add_done_callback(on_complete)
        plt.pause(0.01)  # Keep Matplotlib responsive during background tasks
        return future
    else:
        # Synchronous execution (no threading)
        data = task()
        if callback:
            callback(data)
        return data
 
def close_chart(event):
    plt.pause(0.01)  # Prevent freezing during updates
    plt.close(fig)  # Closes the current figure
# Function to simulate the program loading and update the progress bar
def load_program(progress_bar, window):
    for i in range(101):
        time.sleep(0.01)  # Simulate some work being done
        progress_bar['value'] = i  # Update the progress bar
        window.update_idletasks()  # Update the window
        plt.pause(0.01)  # Keep Matplotlib event loop responsive
    window.after(2, window.destroy)  # Close the window after the progress bar reaches 100%
def load_with_threading():
    window = tk.Tk()
    window.title("Loading InsightFlow...")
    # Create progress bar
 
    progress_bar.pack(pady=20)
    # Start the loading in a background thread
    loading_thread = threading.Thread(target=load_program, args=(progress_bar, window))
    loading_thread.start()
    plt.pause(0.01)  # Keep the Matplotlib event loop responsive
 
    # Run the GUI
    window.mainloop()
def fetch_data_in_background():
    def process_data(data):
        # Callback to process the fetched data
        print("Data fetching completed!")
        # Optionally, add logic to update your charts or UI here.
    # Use the fetch_data_threaded function instead of synchronous fetching
async def fetch_data_in_background_async():
    """
    Asynchronously fetch data in the background and process it.
    """
    data = await asyncio.get_event_loop().run_in_executor(executor, get_data, 'SPY', "1y", "1d")
    process_data(data)
 
def load_notes():
    """Load notes from the file if it exists."""
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "r") as file:
            return file.read()
    return ""
def save_notes(content):
    """Save notes to the file."""
    with open(NOTES_FILE, "w") as file:
        file.write(content)
def show_notes():
    """Open a popup window for the notes."""
    # Create a popup window
    notes_popup = tk.Tk()
    notes_popup.title("Private Commentary")
    notes_popup.geometry("800x600")
    notes_popup.configure(bg="#2c3e50")
    # Add a title label
    title = tk.Label(
        notes_popup,
        text="Notes",
        font=("Helvetica", 16, "bold"),
        fg="#ecf0f1",
        bg="#2c3e50",
        pady=10,
    )
    title.pack()
    # Text box for editing notes
    text_area = tk.Text(
        notes_popup,
        wrap=tk.WORD,
        font=("Courier New", 18),
        bg="#34495e",
        fg="#ecf0f1",
        relief=tk.FLAT,
        padx=10,
        pady=10,
    )
    text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # Load existing notes
    notes_content = load_notes()
    text_area.insert(tk.END, notes_content)
    # Save and close button
    def save_and_close():
        """Save the notes and close the popup."""
        save_notes(text_area.get("1.0", tk.END))  # Save the notes
        notes_popup.destroy()
    save_button = tk.Button(
        notes_popup,
        text="Save & Close",
        command=save_and_close,
        font=("Helvetica", 12),
        bg="#27ae60",
        fg="#ecf0f1",
        relief=tk.FLAT,
        padx=10,
        pady=5,
    )
    save_button.pack(pady=10)
    notes_popup.mainloop()
# Function to create the loading window with a fancy progress bar
 
def load_program(progress_bar, window):
    # Simulate a task with a faster progression (e.g., total_steps reduced or time reduced)
    total_steps = 100
    for step in range(total_steps):
        time.sleep(0.01)  # Decrease the sleep time to speed up the task
        progress_bar['value'] = step + 10  # Update the progress bar value
        window.update_idletasks()  # Update the window without freezing
        window.after(1100, window.destroy)  # Close the window after a brief delay
def show_loading_window():
    # Create a window
    window = tk.Tk()
    window.title("Loading...")
    # Create a label for the popup window
    label = tk.Label(window, text="Connecting with YAHOO, booting up InsightFlow (c) 2024 v6...", font=("Helvetica", 16))
    label.pack(pady=20)
    # Create the progress bar
    progress_bar = ttk.Progressbar(window, style="TProgressbar", length=300, mode="determinate")
    progress_bar.pack(pady=8)
    # Start the loading in a separate thread
    thread = threading.Thread(target=load_program, args=(progress_bar, window))
    thread.start()
    # Start the GUI event loop
    window.mainloop()
# Call the function to show the loading window with the progress bar
show_loading_window()
 
 
 
 
def fetch_and_cache_data(ticker, period, interval, cache_path):
    print("Fetching data from yfinance in background")
    data = yf.download(ticker, period=period, interval=interval)
    print("Columns in downloaded data:", data.columns)
    # Flatten MultiIndex columns if present
    data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
    # Save fetched data to cache
    if not data.empty:
        joblib.dump(data, cache_path)
        print("Data cached successfully.")
    else:
        print(f"No data found for ticker: {ticker}")
def get_data(ticker='SPY', period='5y', interval='1d', use_cache=True):
    # Create the cache directory if it doesn't exist
    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # Define a cache path based on the ticker, period, and interval inside the 'cache' directory
    cache_path = os.path.join(cache_dir, f"cache_{ticker}_{period}_{interval}.pkl")
    # If caching is disabled, fetch data directly from yfinance
    if not use_cache:
        print("Caching is disabled. Fetching data from yfinance")
        data = yf.download(ticker, period=period, interval=interval)
    else:
        # Check if cached data exists
        if os.path.exists(cache_path):
            print("Loading data from cache")
            data = joblib.load(cache_path)  # Load from cache
        else:
            # Fetch data from yfinance in a separate thread
            print("No cache found, fetching data from yfinance in background")
            thread = threading.Thread(target=fetch_and_cache_data, args=(ticker, period, interval, cache_path))
            thread.daemon = True  # Ensure the thread closes when the main program ends
            thread.start()
            thread.join()  # Wait for the thread to complete before proceeding
            # Load data again after the thread completes fetching
            if os.path.exists(cache_path):
                data = joblib.load(cache_path)
            else:
                data = pd.DataFrame()  # Fallback if fetch failed
                
                    # Debugging: Print the fetched data and columns
   # print("Fetched data:")
    #print(data.head())  # Show the first few rows of the data
    #print("Columns in data:", list(data.columns))  # Print column names
    #print("Data columns after processing:", data.columns)
    # Check if the data is empty
    if data.empty:
        print(f"No data found for ticker: {ticker}")
        return pd.DataFrame()  # Return an empty DataFrame to prevent further errors
    return data
  
    
def add_indicators(data, ticker='SPY'):
    # Define the required columns based on the ticker
    close_col = f"Close_{ticker}"
    volume_col = f"Volume_{ticker}"
    high_col = f"High_{ticker}"
    low_col = f"Low_{ticker}"
    # Check for required columns
    if close_col not in data.columns:
        raise KeyError(f"Error: '{close_col}' column not found in data.")
    if volume_col not in data.columns:
        raise KeyError(f"Error: '{volume_col}' column not found in data.")
    if high_col not in data.columns:
        raise KeyError(f"Error: '{high_col}' column not found in data.")
    if low_col not in data.columns:
        raise KeyError(f"Error: '{low_col}' column not found in data.")
    # Moving Averages
    data['short_sma'] = data[close_col].rolling(window=50).mean()
    data['long_sma'] = data[close_col].rolling(window=200).mean()
    data['ema'] = data[close_col].ewm(span=50, adjust=False).mean()
    # Bollinger Bands
    sma = data[close_col].rolling(window=20).mean()
    std = data[close_col].rolling(window=20).std()
    data['upper_band'] = sma + (std * 2)
    data['lower_band'] = sma - (std * 2)
    # RSI (Relative Strength Index)
    delta = data[close_col].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    # MACD (Moving Average Convergence Divergence)
    short_ema = data[close_col].ewm(span=12, adjust=False).mean()
    long_ema = data[close_col].ewm(span=26, adjust=False).mean()
    data['macd'] = short_ema - long_ema
    data['signal_line'] = data['macd'].ewm(span=9, adjust=False).mean()
    # Ichimoku Cloud
    high9 = data[high_col].rolling(window=9).max()
    low9 = data[low_col].rolling(window=9).min()
    data['tenkan_sen'] = (high9 + low9) / 2
    high26 = data[high_col].rolling(window=26).max()
    low26 = data[low_col].rolling(window=26).min()
    data['kijun_sen'] = (high26 + low26) / 2
    high52 = data[high_col].rolling(window=52).max()
    low52 = data[low_col].rolling(window=52).min()
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
    data['senkou_span_b'] = ((high52 + low52) / 2).shift(26)
    data['chikou_span'] = data[close_col].shift(-26)
    # VWMA (Volume-Weighted Moving Average)
    typical_price = (data[high_col] + data[low_col] + data[close_col]) / 3
    data['vwma'] = (typical_price * data[volume_col]).rolling(window=20).sum() / data[volume_col].rolling(window=20).sum()
    data['parabolic_sar'] = data[low_col].rolling(window=2).min()
    # ADX (Average Directional Index)
    high = data[high_col]
    low = data[low_col]
    close = data[close_col]
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr_smooth = tr.rolling(window=14).sum()
    plus_dm_smooth = plus_dm.rolling(window=14).sum()
    minus_dm_smooth = minus_dm.rolling(window=14).sum()
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    data['adx'] = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    # Stochastic Oscillator
    low14 = data[low_col].rolling(window=14).min()
    high14 = data[high_col].rolling(window=14).max()
    data['stochastic'] = 100 * (data[close_col] - low14) / (high14 - low14)
    return data
# Create the price and volume chart with indicators
 
def update_figsize(event):
    global fig, ax1  # Access the global ax1
    if fig is not None:
        # Get the current window size
        width, height = event.width, event.height
        min_width, min_height = 1024, 840
        width = max(width, min_width)
        height = max(height, min_height)
        # Set a dynamic figsize based on the current window size
        fig.set_size_inches(width / fig.dpi, height / fig.dpi)
        plt.draw()  # Redraw the figure with the new figsize
         # cursor = Cursor(ax1, useblit=True, color='white', linewidth=1)  # Reinitialize the cursor
        plt.pause(0.01)  # Keep the Matplotlib event loop responsive
    else:
        print("Figure is not initialized yet.")
     
 
def show_help(event):
    # Create the main popup window
    popup = tk.Tk()
    popup.title("Technical Indicators Help")
    popup.geometry("800x600")  # Set a fixed size for the window
    popup.configure(bg="#2c3e50")  # Dark theme background
    # Add a title label
    title = tk.Label(
        popup,
        text="Technical Indicators Help Guide",
        font=("Helvetica", 16, "bold"),
        fg="#ecf0f1",  # Light text color
        bg="#2c3e50",  # Background matches popup
        pady=10  # Add padding
    )
    title.pack()
    # Add a scrolled text widget for the help content
    text_area = scrolledtext.ScrolledText(
        popup,
        wrap=tk.WORD,  # Wrap text by word
        font=("Courier New", 24),  # Monospaced font for clean look
        bg="#34495e",  # Slightly lighter background
        fg="#ecf0f1",  # Light text color
        relief=tk.FLAT,  # Flat border for modern style
        padx=10,  # Add padding inside the text box
        pady=10
    )
    text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # Add help content
    help_text = (
        "1. Simple Moving Average (SMA) / Exponential Moving Average (EMA):\n"
        "   - **SMA**: The average of the closing prices over a specified period.\n"
        "     It smoothens the price data to identify trends.\n"
        "     Formula: SMA = (Sum of closing prices over N periods) / N\n"
        "   - **EMA**: Similar to SMA but gives more weight to recent prices,\n"
        "     making it more responsive to new information.\n"
        "     Formula: EMA(t) = (Price(t) * α) + EMA(t-1) * (1 - α), where α is the smoothing factor.\n\n"
        
        "1a. 50-Day Moving Average (MA50):\n"
        "   - **MA50**: A specific SMA calculated over 50 days. It is widely used to identify medium-term trends.\n"
        "   - **Usage**: If the price is above the MA50, it may signal an uptrend. Conversely, if the price is below\n"
        "     the MA50, it may indicate a downtrend.\n"
        "   - **Key Insight**: Crossovers between the MA50 and other moving averages (e.g., MA200) can signal major\n"
        "     trend changes, such as the 'Golden Cross' (bullish) or 'Death Cross' (bearish).\n\n"
        "2. Parabolic Stop and Reverse (Parabolic SAR):\n"
        "   - **Definition**: A trend-following indicator that places points on the chart to signal potential reversals.\n"
        "   - **Formula**: Uses a factor (AF) that accelerates the indicator as the trend continues:\n"
        "     SAR(t+1) = SAR(t) + AF * (EP - SAR(t)), where:\n"
        "       - SAR(t): Current SAR value.\n"
        "       - AF: Acceleration Factor (typically starts at 0.02 and increases up to 0.2).\n"
        "       - EP: Extreme Point of the trend (highest high or lowest low).\n"
        "   - **Usage**: Dots appear below the price in an uptrend and above the price in a downtrend. When the price\n"
        "     crosses the SAR dots, it signals a potential trend reversal.\n"
        "   - **Key Insight**: Parabolic SAR works best in trending markets and is less effective in sideways movements.\n\n"
        
        
        "3. Bollinger Bands:\n"
        "   - Consists of three lines: the SMA (middle band), an upper band, and a lower band.\n"
        "   - The upper and lower bands are calculated as: SMA ± (2 * Standard Deviation).\n"
        "   - **Usage**: Measures market volatility. When prices approach the bands, it may signal\n"
        "     overbought (upper band) or oversold (lower band) conditions.\n\n"
        
        "4. Relative Strength Index (RSI):\n"
        "   - A momentum oscillator that measures the speed and change of price movements.\n"
        "   - RSI ranges from 0 to 100.\n"
        "   - **Interpretation**:\n"
        "     - RSI > 70: Overbought (potential reversal or pullback).\n"
        "     - RSI < 30: Oversold (potential upward movement).\n"
        "   - Formula: RSI = 100 - [100 / (1 + RS)], where RS = Average Gain / Average Loss.\n\n"
        
        "5. Moving Average Convergence Divergence (MACD):\n"
        "   - A trend-following momentum indicator showing the relationship between\n"
        "     two moving averages (typically 12-day EMA and 26-day EMA).\n"
        "   - The MACD line is the difference between the two EMAs, while the Signal Line\n"
        "     is a 9-day EMA of the MACD.\n"
        "   - **Usage**: Crossovers and divergences are key signals:\n"
        "     - MACD crossing above Signal Line: Bullish signal.\n"
        "     - MACD crossing below Signal Line: Bearish signal.\n\n"
        
        "6. Average Directional Index (ADX):\n"
        "   - Measures the strength of a trend, ranging from 0 to 100.\n"
        "   - A higher ADX value indicates a stronger trend (uptrend or downtrend).\n"
        "   - **Thresholds**:\n"
        "     - ADX < 25: Weak or no trend.\n"
        "     - ADX > 25: Strong trend.\n\n"
        
        "7. Stochastic Oscillator:\n"
        "   - Measures the current closing price relative to the price range over a specific period.\n"
        "   - Ranges from 0 to 100.\n"
        "   - **Interpretation**:\n"
        "     - Above 80: Overbought (possible downward correction).\n"
        "     - Below 20: Oversold (possible upward correction).\n"
        "   - Formula: %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100.\n\n"
        
        "8. Ichimoku Cloud:\n"
        "   - A comprehensive indicator that shows support/resistance levels, trend direction, momentum,\n"
        "     and potential reversal points.\n"
        "   - Key Components:\n"
        "     - Tenkan-sen (Conversion Line): (9-period High + 9-period Low) / 2.\n"
        "     - Kijun-sen (Base Line): (26-period High + 26-period Low) / 2.\n"
        "     - Senkou Span A: (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods forward.\n"
        "     - Senkou Span B: (52-period High + 52-period Low) / 2, shifted 26 periods forward.\n"
        "     - Chikou Span (Lagging Line): Today's closing price, plotted 26 periods back.\n"
        "   - **Usage**: The area between Senkou Span A and B forms the 'cloud,'\n"
        "     indicating support/resistance.\n\n"
        
        "9. Volume-Weighted Moving Average (VWMA):\n"
        "   - A moving average that gives more weight to periods with higher trading volume.\n"
        "   - Formula: VWMA = Σ(Price * Volume) / Σ(Volume).\n"
        "   - **Usage**: Highlights price levels where trading activity is concentrated.\n\n"
        
        "(c) 2024 SIG LABS - Proprietary Technical Analytics Software v6\n"
        "All rights reserved. Unauthorized distribution is prohibited.\n"
    )
    
    # Insert the help text into the text area
    text_area.insert(tk.END, help_text)
    text_area.configure(state='disabled')  # Make text read-only
    # Add a Close button
    close_button = tk.Button(
        popup,
        text="Close",
        command=popup.destroy,
        font=("Helvetica", 12),
        bg="#e74c3c",  # Red button
        fg="#ecf0f1",  # White text
        relief=tk.FLAT,
        padx=10,
        pady=5
    )
    close_button.pack(pady=10)
    # Run the popup window
    popup.mainloop()
    plt.pause(0.01)  # Keep the Matplotlib event loop responsive
    
def create_chart(data):
    """
    Create a detailed chart with subplots for price, volume, and various indicators,
    including buttons to toggle visibility of signals and other interactive elements.
    Parameters:
        data (pd.DataFrame): The data containing stock prices, volume, and indicators.
    """
    global fig, ax1, ax2, ax_rsi, ax_macd, ax_adx, ax_stochastic
    global line_close, line_short_sma, line_long_sma, line_ema
    global line_upper_bb, line_lower_bb, line_chikou, scatter_sar
    global cloud_bullish, cloud_bearish
    # Initialize figure and subplots
    fig = plt.figure(figsize=(16, 20))
    fig.canvas.manager.set_window_title('InsightFlow 6 Private Investment Software  (c) 2024 Peter De Ceuster SIG LABS')
    fig.canvas.mpl_connect('resize_event', update_figsize)
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(0.7)
    # Maximize the window
    manager = plt.get_current_fig_manager()
    try:
        manager.window.state('zoomed')
    except AttributeError:
        try:
            manager.window.showMaximized()
        except AttributeError:
            print("Maximizing not supported for this backend.")
    # Buttons for Notes and Target Analysis
    button_notes_ax = fig.add_axes([0.22, 0.95, 0.08, 0.03])
    notes_button = Button(button_notes_ax, 'NOTES', color='#3498db', hovercolor='#2980b9')
    notes_button.on_clicked(lambda event: show_notes())
    button_target_ax = fig.add_axes([0.32, 0.95, 0.08, 0.03])
    target_button = Button(button_target_ax, 'Analyse Asset', color='#3498db', hovercolor='#2980b9')
    target_button.on_clicked(launch_target)
    # Create subplots using gridspec
    gs = fig.add_gridspec(5, 1, height_ratios=[4, 1, 1, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = ax1.twinx()
    ax_rsi = fig.add_subplot(gs[1], sharex=ax1)
    ax_macd = fig.add_subplot(gs[2], sharex=ax1)
    ax_adx = fig.add_subplot(gs[3], sharex=ax1)
    ax_stochastic = fig.add_subplot(gs[4], sharex=ax1)
    # Set subplot appearance
    for ax in [ax1, ax2, ax_rsi, ax_macd, ax_adx, ax_stochastic]:
        ax.set_facecolor('black')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
    # Main Price and Volume Plot
    line_close, = ax1.plot(data.index, data['Close_SPY'], color='tab:blue', label='Closing Price')
    line_short_sma, = ax1.plot(data.index, data['short_sma'], color='purple', linestyle='--', label='50-Day SMA')
    line_long_sma, = ax1.plot(data.index, data['long_sma'], color='orange', linestyle='--', label='200-Day SMA')
    line_ema, = ax1.plot(data.index, data['ema'], color='brown', linestyle='-', label='EMA (50)')
    line_upper_bb, = ax1.plot(data.index, data['upper_band'], color='red', linestyle='dotted', label='Upper Bollinger Band')
    line_lower_bb, = ax1.plot(data.index, data['lower_band'], color='green', linestyle='dotted', label='Lower Bollinger Band')
    # Volume Background for Main Chart
    # Separate Up and Down Volume Bars (Green for Up, Red for Down)
    up_volume = data[data['Close_SPY'] >= data['Open_SPY']]  # Up Volume when Close > Open
    down_volume = data[data['Close_SPY'] < data['Open_SPY']]  # Down Volume when Close < Open
    
    # Plot Up Volume Bars in Green
    ax2.bar(up_volume.index, up_volume['Volume_SPY'], color='green', alpha=0.6, label='Up Volume')
    
    # Plot Down Volume Bars in Red
    ax2.bar(down_volume.index, down_volume['Volume_SPY'], color='red', alpha=0.6, label='Down Volume')
    
    # Optional: Add a legend for Volume
    ax2.legend(loc='upper left', fontsize=8, facecolor='lightblue', edgecolor='white')
    
    # Ichimoku Cloud
    cloud_bullish = ax1.fill_between(data.index, data['senkou_span_a'], data['senkou_span_b'],
                                     where=(data['senkou_span_a'] > data['senkou_span_b']),
                                     color='green', alpha=0.3, label='Ichimoku Bullish Cloud')
    cloud_bearish = ax1.fill_between(data.index, data['senkou_span_a'], data['senkou_span_b'],
                                     where=(data['senkou_span_a'] <= data['senkou_span_b']),
                                     color='red', alpha=0.3, label='Ichimoku Bearish Cloud')
    # Scatter Points (e.g., Parabolic SAR)
    scatter_sar = ax1.scatter(data.index, data['parabolic_sar'], color='yellow', label='Parabolic SAR', s=10)
  # Add Close Price and Volume Background to Lower Subplots with Secondary Axis
    # Add Close Price and Volume Background to Lower Subplots with Secondary Axis
    for lower_ax in [ax_rsi, ax_macd, ax_adx, ax_stochastic]:
        # Plot Close Price on the primary axis
        lower_ax.plot(data.index, data['Close_SPY'], color='orange', linestyle='--', label='Close Price')
    
        # Create a secondary y-axis for volume
        lower_ax_volume = lower_ax.twinx()
        
        # Separate Up and Down Volume Bars (Green for Up, Red for Down)
        up_volume = data[data['Close_SPY'] >= data['Open_SPY']]  # Up Volume when Close > Open
        down_volume = data[data['Close_SPY'] < data['Open_SPY']]  # Down Volume when Close < Open
    
        # Plot Up Volume Bars in Green
        lower_ax_volume.bar(up_volume.index, up_volume['Volume_SPY'], color='blue', alpha=0.5)
    
        # Plot Down Volume Bars in Red
        lower_ax_volume.bar(down_volume.index, down_volume['Volume_SPY'], color='orange', alpha=0.5)
    
        # Set volume axis properties for clarity
        lower_ax_volume.set_ylim(0, data['Volume_SPY'].max() * 1.1)  # Scale slightly above max
        lower_ax_volume.set_yticks([])  # Remove ticks to avoid clutter
    
    #print("Close Price values:")
    #print(data['Close_SPY'].head())
    #print("RSI values:")
    #print(data['rsi'].head())
    #print("Stochastic values:")
    #print(data['stochastic'].head())
    #print("ADX values:")
    #print(data['adx'].head())
 
    
    # RSI Plot
    ax_rsi.plot(data.index, data['rsi'], color='blue', label='RSI')
    ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax_rsi.set_ylim(0, 100)  # Explicit scaling for RSI
    ax_rsi_secondary = ax_rsi.twinx()  # Add secondary axis for Close Price
    ax_rsi_secondary.plot(data.index, data['Close_SPY'], color='yellow', linestyle='-', linewidth=1.2, label='Close Price')
    ax_rsi.legend(loc='upper left', fontsize=8, facecolor='lightblue', edgecolor='white')
    ax_rsi.set_title('Relative Strength Index (RSI)')  # Title for RSI
    
    # Step 1: Calculate MACD and Signal Line (Drastic Calculation Approach)
    # Calculate the 12-period and 26-period Exponential Moving Averages (EMA)
    short_ema = data['Close_SPY'].ewm(span=12, adjust=False).mean()  # Short EMA (12 periods)
    long_ema = data['Close_SPY'].ewm(span=26, adjust=False).mean()  # Long EMA (26 periods)
    
    # Calculate MACD as the difference between the short and long EMA
    data['macd'] = short_ema - long_ema  # MACD Line
    
    # Calculate the Signal Line as the 9-period EMA of the MACD
    data['signal_line'] = data['macd'].ewm(span=9, adjust=False).mean()  # Signal Line
    
    # Step 2: Verify the MACD and Signal Line calculations with debug output
    #print("MACD Head:\n", data['macd'].head())  # Check MACD values
    #print("Signal Line Head:\n", data['signal_line'].head())  # Check Signal Line values
    
    # Check the difference between MACD and Signal Line to verify
    data['macd_signal_diff'] = data['macd'] - data['signal_line']
    #print("MACD vs Signal Line Difference Head:\n", data['macd_signal_diff'].head())  # Should not be zero
    
    # Step 3: Plot MACD, Signal Line, and Close Price on the secondary axis
    # MACD Plot
    ax_macd.plot(data.index, data['macd'], color='blue', label='MACD')  # MACD line in blue
    ax_macd.plot(data.index, data['signal_line'], color='red', linestyle='--', label='Signal Line')  # Signal Line in red
    ax_macd.set_title('Moving Average Convergence Divergence (MACD)')  # Title for MACD
    
    # Adjust y-limits for the MACD plot to ensure both MACD and Signal Line fit
    ax_macd.set_ylim(min(data['macd'].min(), data['signal_line'].min()) - 10, max(data['macd'].max(), data['signal_line'].max()) + 10)
    
    # Create a secondary axis for Close Price
    ax_macd_secondary = ax_macd.twinx()  # Create a secondary axis for Close Price
    ax_macd_secondary.plot(data.index, data['Close_SPY'], color='yellow', linestyle='-', linewidth=1.2, label='Close Price')
    
    # Legends for MACD Plot
    ax_macd.legend(loc='upper left', fontsize=8, facecolor='lightblue', edgecolor='white')
    
        # Add a legend to the main graph (the one with all signals and buttons)
    ax1.set_title('Professional Chart')  # Title for main plot
    ax1.legend(loc='upper left', fontsize=15, facecolor='lightblue', edgecolor='white', 
               bbox_to_anchor=(1.05, 1))  # Move legend outside to the right side, above the buttons
               
    
    
    
 
    
    
    
    
    
    
    
    
    
    
    # ADX Plot
    ax_adx.plot(data.index, data['adx'], color='purple', label='ADX')
    ax_adx.axhline(25, color='red', linestyle='--', label='Strong Trend (>25)')
    ax_adx.set_ylim(0, 100)  # Explicit scaling for ADX
    ax_adx_secondary = ax_adx.twinx()  # Add secondary axis for Close Price
    ax_adx_secondary.plot(data.index, data['Close_SPY'], color='yellow', linestyle='-', linewidth=1.2, label='Close Price')
    ax_adx.legend(loc='upper left', fontsize=8, facecolor='lightblue', edgecolor='white')
    ax_adx.set_title('Average Directional Index (ADX)')  # Title for ADX
    
    # Stochastic Plot
    ax_stochastic.plot(data.index, data['stochastic'], color='white', label='Stochastic Oscillator')
    ax_stochastic.axhline(80, color='red', linestyle='--', label='Overbought (80)')
    ax_stochastic.axhline(20, color='green', linestyle='--', label='Oversold (20)')
    ax_stochastic.set_ylim(0, 100)  # Explicit scaling for Stochastic
    ax_stochastic_secondary = ax_stochastic.twinx()  # Add secondary axis for Close Price
    ax_stochastic_secondary.plot(data.index, data['Close_SPY'], color='yellow', linestyle='-', linewidth=1.2, label='Close Price')
    ax_stochastic.legend(loc='upper left', fontsize=8, facecolor='lightblue', edgecolor='white')
    ax_stochastic.set_title('Stochastic Oscillator')  # Title for Stochastic Oscillator
    
    
    
    
    
    
    # Toggle Buttons for Main Signals
    signals = [
        (line_close, "Close Price"),
        (line_short_sma, "50-Day SMA"),
        (line_long_sma, "200-Day SMA"),
        (line_ema, "EMA (50)"),
        (line_upper_bb, "Upper Bollinger Band"),
        (line_lower_bb, "Lower Bollinger Band"),
        (scatter_sar, "Parabolic SAR"),
        (cloud_bullish, "Ichimoku Bull"),
        (cloud_bearish, "Ichimoku Bear")
    ]
    buttons = []
    for idx, (signal, label) in enumerate(signals):
        btn_ax = fig.add_axes([0.85, 0.7 - idx * 0.05, 0.1, 0.04])  # Dynamically position buttons
        button = Button(btn_ax, label)
        
        # Callback for toggling visibility
        def toggle_visibility(event, sig=signal):
            sig.set_visible(not sig.get_visible())
            fig.canvas.draw_idle()
        button.on_clicked(toggle_visibility)
        buttons.append(button)
        
            # Custom Help Button
    button_ax = fig.add_axes([0.02, 0.95, 0.08, 0.03])
    help_button = Button(button_ax, 'HELP', color='#27ae60', hovercolor='#1abc9c')
    help_button.on_clicked(show_help)
    
    
    
       # Close Button
    close_button_ax = fig.add_axes([0.12, 0.95, 0.08, 0.03])
    close_button = Button(close_button_ax, 'CLOSE', color='#e74c3c', hovercolor='#c0392b')
    close_button.on_clicked(close_chart)
    
    
    # Adjust Layout
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.8, hspace=0.5)
    cursor = mplcursors.cursor(line_close, hover=True)
    # Change the color of all plot titles at once
    for ax in [ax1, ax_rsi, ax_macd, ax_adx, ax_stochastic]:
        ax.set_title(ax.get_title(), color='#DC143C')  # Set all titles to 'darkblue'
        crosshair = Cursor(ax1, useblit=True, color='red', linewidth=1)
    plt.show()
 
 
    
    
async def async_fetch_and_process_data(ticker, period, interval):
    """
    Asynchronously fetch and process data using ThreadPoolExecutor to offload blocking tasks.
    """
    print("Fetching data asynchronously...")
    loop = asyncio.get_event_loop()
    # Offload blocking data fetching to ThreadPoolExecutor
    data = await loop.run_in_executor(executor, get_data, ticker, period, interval)
    # Add indicators (this is also CPU-bound but runs after fetching)
    data = await loop.run_in_executor(executor, add_indicators, data)
    return data
async def async_main():
    """
    Main entry point for the program with async functionality.
    """
    global fig  # Use the global variable
    ticker = 'SPY'
    try:
        # Fetch and process data asynchronously
        data = await async_fetch_and_process_data(ticker, period="5y", interval="1d")
        print("Data fetched successfully. (c) 2024 Peter De Ceuster all rights reserved  v 6.0")
        # Create the chart (can be made async if necessary)
        create_chart(data)
    except KeyError as e:
        #print(f"Column missing error detected: {e}. Restarting program.")
        print(f"Rehashing new Ticker Data, hold on. Restarting InsightFlow 6. (c) SIG LABS 2024")
        restart_program()
    except Exception as e:
        print(f"Error during data fetching or processing: {e}")
    finally:
        # Shutdown the executor to release resources
        executor.shutdown(wait=True)
        
        
        
if __name__ == "__main__":
    validate_and_clean_cache()
    asyncio.run(async_main())