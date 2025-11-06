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
spacing = 400  # spacing within each montage pair
spacing_cluster = 800  # spacing between montage clusters
limit = 500

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
        
        # 시간 범위 필터링
        time_array = df[time_col].values
        
        if start_time is not None and end_time is not None:
            mask = (time_array >= start_time) & (time_array <= end_time)
            df_filtered = df[mask].reset_index(drop=True)
            print(f"  Filtered time range: {start_time}s to {end_time}s")
        else:
            df_filtered = df
            start_time = time_array[0]
            end_time = time_array[-1]
            print(f"  Using full time range: {start_time:.2f}s to {end_time:.2f}s")
        
        # Define the montage groups (bipolar montage)
        # Each group is a list of electrode pairs: [(top, bottom), (top, bottom), ...]
        montage_groups = [
            [('C1', 'P3'), ('P3', 'T5'), ('T5', 'O1')],  # Group 1
            [('C1', 'Pz'), ('Pz', 'O1')],                # Group 2
            [('C2', 'Pz'), ('Pz', 'O2')],                # Group 3
            [('C2', 'P4'), ('P4', 'T6'), ('T6', 'O2')],  # Group 4
        ]
        
        # Filter available channels
        available_channels = set(df_filtered.columns) - {time_col}
        
        # Verify all channels in montage groups are available
        all_montage_channels = set()
        for group in montage_groups:
            for ch1, ch2 in group:
                all_montage_channels.add(ch1)
                all_montage_channels.add(ch2)
        
        missing_channels = all_montage_channels - available_channels
        
        # Count total montage pairs
        total_pairs = sum(len(group) for group in montage_groups)
        
        if missing_channels:
            print(f"  Warning: Missing channels: {missing_channels}")
        
        if total_pairs > 0:
            print(f"  Plotting {total_pairs} montage pairs from {len(montage_groups)} groups")
            print(f"  Pair spacing: {spacing} uV, Cluster spacing: {spacing_cluster} uV")
            
            # 화면 사이즈에 맞춰서 figure 크기 설정
            height_per_pair = screen_height_inch * 0.8  # 화면 높이의 80% 사용
            fig_height = max(height_per_pair, 4)  # 최소 4인치
            
            fig, ax = plt.subplots(figsize=(screen_width_inch * 0.95, fig_height))
            
            time_filtered = df_filtered[time_col].values
            # Convert time to relative time (0 to duration)
            time_relative = time_filtered - start_time
            
            # y축 ticks와 labels 설정용
            y_ticks = []
            y_labels = []
            
            # Calculate pairs per group first
            group_pairs_info = []  # List of (group_idx, pairs_in_group)
            total_montage_pairs = 0
            
            for group_idx, group in enumerate(montage_groups):
                pairs_in_this_group = 0
                for ch1, ch2 in group:
                    if ch1 in available_channels and ch2 in available_channels:
                        pairs_in_this_group += 1
                        total_montage_pairs += 1
                if pairs_in_this_group > 0:
                    group_pairs_info.append((group_idx, pairs_in_this_group))
            
            # Plot each montage group (from top to bottom)
            current_y_offset = total_montage_pairs * spacing + (len(group_pairs_info) - 1) * spacing_cluster
            
            for info_idx, (group_idx, num_pairs_in_group) in enumerate(group_pairs_info):
                group = montage_groups[group_idx]
                
                for pair_idx_in_group, (ch1, ch2) in enumerate(group):
                    # Check if channels exist
                    if ch1 not in available_channels or ch2 not in available_channels:
                        continue
                    
                    # Get data for both channels
                    data_ch1 = df_filtered[ch1].values
                    data_ch2 = df_filtered[ch2].values
                    
                    # Calculate montage: ch1 - ch2
                    montage_data = data_ch1 - data_ch2
                    
                    # Plot the montage data (black line)
                    ax.plot(time_relative, montage_data + current_y_offset, color='black', linewidth=0.8)
                    
                    # Save y-axis tick and label
                    y_ticks.append(current_y_offset)
                    y_labels.append(f'{ch1}-{ch2}')
                    
                    # Move down for next pair
                    current_y_offset -= spacing
                
                # Move down extra for cluster spacing gap (if not last group)
                # The gap between clusters is (spacing_cluster - spacing)
                if info_idx < len(group_pairs_info) - 1:
                    current_y_offset -= (spacing_cluster - spacing)
            
            # Set y-axis ticks and labels
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=9)
            
            # Calculate scale bar position (at top left)
            scale_bar_height = 200  # uV
            scale_bar_x_start = 0.02 * duration
            scale_bar_x_end = scale_bar_x_start
            # Position at the top of the plot (highest y value)
            scale_bar_y_start = y_ticks[0] + 500 if y_ticks else 200  # Place above the top trace (higher)
            scale_bar_y_end = scale_bar_y_start - scale_bar_height
            
            ax.plot([scale_bar_x_start, scale_bar_x_end], 
                   [scale_bar_y_start, scale_bar_y_end], 
                   'r-', linewidth=2)
            
            # Draw gray vertical lines every 1 second across the entire graph
            for t in np.arange(0, duration + 1, 1):
                if t <= duration:
                    if t % 5 == 0:
                        # Every 5 seconds: darker gray and thicker
                        ax.axvline(x=t, color='#1a1a1a', alpha=0.8, linewidth=1.5, linestyle='-')
                    else:
                        # Every 1 second: dark gray with opacity
                        ax.axvline(x=t, color='#404040', alpha=0.4, linewidth=0.8, linestyle='-')
            
            # 축 설정
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Voltage (μV)', fontsize=11)
            ax.set_xlim(0, duration)
            # Y축 범위를 설정 - 모든 데이터가 안에 들어가도록 충분한 margin 추가
            if y_ticks:
                y_min = min(y_ticks) - limit * 1.2  # Extra margin at bottom
                y_max = max(y_ticks) + limit * 1.2  # Extra margin at top
            else:
                y_min = -limit
                y_max = limit
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.2)
            
            plt.tight_layout()
            
            # Save as PNG file with 210:135 aspect ratio and high quality
            png_filename = csv_file.replace('.csv', '.png')
            png_filepath = os.path.join(data_dir, png_filename)
            # 210:135 aspect ratio (approximately 1.56:1)
            fig.set_size_inches(14.0, 9.0)  # width:height ratio = 210:135
            plt.savefig(png_filepath, format='png', dpi=300, bbox_inches='tight')
            print(f"  Saved PNG: {png_filepath}")
            
            plt.show()
            
    except Exception as e:
        print(f"  Error reading {csv_file}: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")
