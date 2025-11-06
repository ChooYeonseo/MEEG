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
duration = 40
end_time =  start_time + duration   # 예: 60
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
            data_cols = [col for col in df.columns if col != 'time']
        else:
            time_col = df.columns[0]
            data_cols = df.columns[1:]
        
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
        
        # 데이터 컬럼이 있으면 플롯
        if len(data_cols) > 0:
            n_channels = len(data_cols)
            
            print(f"  Channel spacing: {spacing:.2f} uV")
            
            # 화면 사이즈에 맞춰서 figure 크기 설정
            # 채널 당 높이 비율 조정
            height_per_channel = screen_height_inch * 0.8  # 화면 높이의 80% 사용
            fig_height = max(height_per_channel, 4)  # 최소 4인치
            
            fig, ax = plt.subplots(figsize=(screen_width_inch * 0.95, fig_height))
            
            time_filtered = df_filtered[time_col].values
            # Convert time to relative time (0 to duration)
            time_relative = time_filtered - start_time
            
            # y축 ticks와 labels 설정용
            y_ticks = []
            y_labels = []
            
            # 각 채널별로 offset을 주면서 플롯
            for idx, col in enumerate(data_cols):
                data = df_filtered[col].values
                # Flip the data (negate it)
                data = -data
                
                # 동일한 간격(200 uV) offset 설정
                offset = idx * spacing
                
                # 데이터 플롯 (검은색)
                ax.plot(time_relative, data + offset, color='black', linewidth=0.8)
                
                # y축 틱과 레이블 저장
                y_ticks.append(offset)
                y_labels.append(col)
            
            # Set y-axis ticks and labels
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=9)
            
            # 200 uV scale bar 그리기 (빨간색) - 왼쪽 위 모서리
            scale_bar_height = 200  # uV
            scale_bar_x_start = 0.02 * duration
            scale_bar_x_end = scale_bar_x_start
            scale_bar_y_start = (n_channels - 1) * spacing + limit - 50  # 상단에 위치
            scale_bar_y_end = scale_bar_y_start - scale_bar_height
            
            ax.plot([scale_bar_x_start, scale_bar_x_end], 
                   [scale_bar_y_start, scale_bar_y_end], 
                   'r-', linewidth=2)
            ax.text(scale_bar_x_start + 0.02 * duration, 
                   (scale_bar_y_start + scale_bar_y_end) / 2, 
                   '200 μV', ha='left', va='center', fontsize=10, fontweight='bold', color='red')
            
            # 축 설정
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Voltage (μV)', fontsize=11)
            ax.set_xlim(0, duration)
            # Y축 범위를 설정하되, 필요하면 확장
            ax.set_ylim(-limit, (n_channels - 1) * spacing + limit)
            ax.grid(True, alpha=0.2)
            
            plt.suptitle(f'{csv_file} ({duration}s)', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()
        else:
            print(f"  Warning: No data columns found")
            
    except Exception as e:
        print(f"  Error reading {csv_file}: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")
