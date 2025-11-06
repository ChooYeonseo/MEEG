import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# 화면 사이즈 자동 감지
import matplotlib
screen_dpi = matplotlib.rcParams['figure.dpi']
screen_width_inch = 18  # 기본값
screen_height_inch = 10  # 기본값

try:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    screen_width_pixel = root.winfo_screenwidth()
    screen_height_pixel = root.winfo_screenheight()
    screen_width_inch = screen_width_pixel / screen_dpi
    screen_height_inch = screen_height_pixel / screen_dpi
    root.destroy()
    print(f"Screen size detected: {screen_width_pixel}x{screen_height_pixel} pixels")
    print(f"Screen size in inches: {screen_width_inch:.1f}x{screen_height_inch:.1f} inches")
except:
    print(f"Using default screen size: {screen_width_inch}x{screen_height_inch} inches")

# CSV 파일이 있는 디렉토리
data_dir = "/Volumes/CHOO'S SSD/LINK/1. 박진봉 교수님 데이터/1st_trial"

# 시작 시간과 끝 시간 설정 (초 단위, None이면 전체 데이터)
start_time = 55  # 예: 0
duration = 35
end_time =  start_time + duration   # 예: 60
# Extra data for STFT padding (to ensure full window analysis at the end)
extra_duration = 1.0
end_time_extended = end_time + extra_duration
spacing = 600
limit = 600

# 디렉토리의 모든 CSV 파일 찾기
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
csv_files.sort()

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {f}")

# 각 CSV 파일을 읽고 플롯
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        
        print(f"\nProcessing: {csv_file}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # 'time' 컬럼이 있는지 확인
        if 'time' in df.columns:
            time_col = 'time'
        else:
            time_col = df.columns[0]
        
        # Define the desired channel order
        desired_order = ['Fp1', 'F3', 'P3', 'T5', 'O1', 'Fp2', 'F4', 'P4', 'T6', 'O2', 'C1', 'P1', 'C2', 'P2', 'Fz', 'Pz']
        
        # Filter available channels in the desired order
        data_cols = [col for col in desired_order if col in df.columns]
        
        # 시간 범위 필터링 (including extra data for STFT window analysis)
        time_array = df[time_col].values
        
        if start_time is not None and end_time is not None:
            # Use extended end time to get extra data for proper STFT windowing
            mask = (time_array >= start_time) & (time_array <= end_time_extended)
            df_filtered = df[mask].reset_index(drop=True)
            print(f"  Filtered time range: {start_time}s to {end_time_extended}s (extended for STFT)")
            print(f"  Actual data points: {len(df_filtered)}")
            if len(df_filtered) > 0:
                actual_start = time_array[mask][0]
                actual_end = time_array[mask][-1]
                actual_duration = actual_end - actual_start
                print(f"  Actual time range in data: {actual_start:.4f}s to {actual_end:.4f}s")
                print(f"  Actual duration: {actual_duration:.4f}s (requested: {duration}s)")
        else:
            df_filtered = df
            start_time = time_array[0]
            end_time = time_array[-1]
            print(f"  Using full time range: {start_time:.2f}s to {end_time:.2f}s")
        
        # 데이터 컬럼이 있으면 STFT 플롯
        if len(data_cols) > 0:
            n_channels = len(data_cols)
            
            print(f"  Channels to plot: {data_cols}")
            
            # Calculate average data across all channels
            time_filtered = df_filtered[time_col].values
            time_relative = time_filtered - start_time
            
            # Extract all channel data and compute average
            channel_data_list = []
            for col in data_cols:
                channel_data_list.append(df_filtered[col].values)
            
            # Average across all channels
            average_data = np.mean(channel_data_list, axis=0)
            
            print(f"  Average data shape: {average_data.shape}")
            
            # Get sampling rate (assuming it's the time difference between consecutive samples)
            if len(time_filtered) > 1:
                sampling_rate = 1.0 / np.mean(np.diff(time_filtered))
                print(f"  Estimated sampling rate: {sampling_rate:.2f} Hz")
            else:
                sampling_rate = 1000  # Default value
                print(f"  Using default sampling rate: {sampling_rate} Hz")
            
            # Compute STFT with 1 Hz frequency resolution and 1 sec time resolution
            # For 1 Hz resolution: nperseg = sampling_rate / 1
            nperseg = int(sampling_rate / 1.0)
            nperseg = min(nperseg, len(average_data))
            
            # Define power range
            vmin = 0
            vmax = 30
            
            # Use 50% overlap (noverlap = nperseg // 2) for smooth continuous STFT
            frequencies, times, Sxx = signal.spectrogram(
                average_data, 
                fs=sampling_rate, 
                nperseg=nperseg,
                noverlap=nperseg // 2,
                window='hann'
            )
            
            # Filter to 0-30 Hz range
            freq_mask = frequencies <= 30
            frequencies_filtered = frequencies[freq_mask]
            Sxx_filtered = Sxx[freq_mask, :]
            
            # Convert to dB scale for better visualization
            Sxx_db = 10 * np.log10(Sxx_filtered + 1e-10)
            
            # Shift times to start from 0 (relative to start_time)
            times_relative = times - times[0]
            
            # Create figure with 230:18 aspect ratio
            # 230:18 = 12.78:1
            fig_width = 12.78
            fig_height = 1.0
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Plot STFT with vmin and vmax using relative times
            im = ax.pcolormesh(times_relative, frequencies_filtered, Sxx_db, 
                               shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_ylim([0, 30])
            ax.set_xlim([times_relative[0], times_relative[-1]])
            
            # Show ticks and x-axis labels, hide y-axis labels
            ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=True)
            
            # Set frequency ticks every 5 Hz
            freq_ticks = np.arange(0, 35, 5)
            ax.set_yticks(freq_ticks)
            
            # Set time ticks every 5 seconds from 0 to duration
            time_ticks = np.arange(0, duration + 1, 5)
            ax.set_xticks(time_ticks)
            # Format tick labels to show duration (0 to 35)
            time_labels = [f'{int(t)}' for t in time_ticks]
            ax.set_xticklabels(time_labels)
            
            # Remove x and y axis labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            plt.tight_layout()
            
            # Save as PNG file with 230:18 aspect ratio and high quality
            png_filename = csv_file.replace('.csv', '_STFT.png')
            png_filepath = os.path.join(data_dir, png_filename)
            # 230:18 aspect ratio
            fig.set_size_inches(12.78, 1.0)
            plt.savefig(png_filepath, format='png', dpi=300, bbox_inches='tight')
            print(f"  Saved STFT PNG: {png_filepath}")
            
            plt.show()
            
            # Create horizontal power bar (colorbar as separate figure)
            fig_bar, ax_bar = plt.subplots(figsize=(12.78, 0.5))
            ax_bar.axis('off')
            
            # Create a colorbar as a separate image
            cbar_ax = fig_bar.add_axes([0.1, 0.3, 0.8, 0.4])
            cb = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cb.set_label('')
            # Set colorbar ticks every 10 dB
            cbar_ticks = np.arange(vmin, vmax + 10, 10)
            cb.set_ticks(cbar_ticks)
            cb.ax.tick_params(labelbottom=False)
            
            # Save colorbar as separate PNG
            bar_filename = csv_file.replace('.csv', '_PowerBar.png')
            bar_filepath = os.path.join(data_dir, bar_filename)
            fig_bar.set_size_inches(12.78, 0.5)
            plt.savefig(bar_filepath, format='png', dpi=300, bbox_inches='tight')
            print(f"  Saved Power Bar PNG: {bar_filepath}")
            
            plt.show()
        else:
            print(f"  Warning: No data columns found")
            
    except Exception as e:
        print(f"  Error reading {csv_file}: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")
