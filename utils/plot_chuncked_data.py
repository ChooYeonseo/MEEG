import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from scipy import signal
from scipy.stats import zscore
# EEG 데이터 시각화 함수
def plot_eeg_data(data, start_time=10, end_time=20, figsize=(15, 12), tick_scale=100, buffer_scale=50):
    """
    지정된 시간 범위의 EEG 데이터를 subplot으로 시각화
    
    Parameters:
    -----------
    data : pandas.DataFrame
        EEG 데이터 (time 컬럼과 pin 컬럼들 포함)
    start_time : float
        시작 시간 (초)
    end_time : float  
        종료 시간 (초)
    figsize : tuple
        그래프 크기
    """
    # 시간 범위로 데이터 필터링
    time_mask = (data['time'] >= start_time) & (data['time'] <= end_time)
    filtered_data = data[time_mask].copy()
    
    if len(filtered_data) == 0:
        print(f"Warning: No data found in time range {start_time}-{end_time} seconds")
        return
    
    # pin 컬럼들만 추출 (time 제외)
    pin_columns = [col for col in data.columns if col != 'time']
    n_channels = len(pin_columns)
    
    # subplot 레이아웃 계산 (세로로 배치)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    
    # 채널이 1개인 경우 axes를 리스트로 만들기
    if n_channels == 1:
        axes = [axes]
    
    # 모든 채널의 데이터 범위 계산 (동일한 y축 범위 적용)
    all_data = []
    for pin_name in pin_columns:
        all_data.extend(filtered_data[pin_name].values)
    
    all_data = np.array(all_data)
    global_max = np.max(all_data)
    global_min = np.min(all_data)
    
    y_min = np.floor(global_min / buffer_scale) * buffer_scale  # global_min 보다 작은 값중 50의 배수
    y_max = np.ceil(global_max / buffer_scale) * buffer_scale  # global_max 보다 큰 값중 50의 배수
    
    # 각 채널별로 subplot 그리기
    for i, pin_name in enumerate(pin_columns):
        axes[i].plot(filtered_data['time'], filtered_data[pin_name], 
                    linewidth=0.8, color=f'C{i}')
        
        axes[i].set_ylabel(f'{pin_name}\n(µV)', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'EEG Signal - {pin_name}', fontsize=11, pad=5)
        
        # 모든 채널에 동일한 y축 범위 적용
        axes[i].set_ylim(y_min, y_max)
        # y축 틱을 100µV 간격으로 설정 (100의 배수만)
        tick_start = np.ceil(y_min / tick_scale) * tick_scale
        tick_end = np.floor(y_max / tick_scale) * tick_scale
        y_ticks = np.arange(tick_start, tick_end + tick_scale, tick_scale)
        axes[i].set_yticks(y_ticks)
        # x축 범위를 정확히 맞춤
        axes[i].set_xlim(start_time, end_time)
    
    # x축 레이블은 마지막 subplot에만
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    
    # 전체 제목
    fig.suptitle(f'EEG Signals ({start_time}-{end_time} seconds)', 
                fontsize=14, fontweight='bold')
    
    # 레이아웃 조정
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    # 통계 정보 출력
    print(f"📊 Data Range Summary ({start_time}-{end_time}s):")
    print(f"   - Total samples: {len(filtered_data):,}")
    print(f"   - Sampling rate: ~{len(filtered_data)/(end_time-start_time):.0f} Hz")
    print(f"   - Channels: {pin_columns}")


# 다른 시간 구간도 테스트해보기
# plot_eeg_data(data, start_time=0, end_time=5)
# plot_eeg_data(data, start_time=50, end_time=60)


# 빠른 시각화를 위한 프리셋 함수들
def plot_first_30s(data):
    """첫 30초 데이터 시각화"""
    plot_eeg_data(data, 0, min(30, data['time'].max()))

def plot_middle_section(data):
    """중간 섹션 데이터 시각화"""
    total_duration = data['time'].max()
    mid_point = total_duration / 2
    start = max(0, mid_point - 15)
    end = min(total_duration, mid_point + 15)
    plot_eeg_data(data, start, end)

def plot_last_30s(data):
    """마지막 30초 데이터 시각화"""
    total_duration = data['time'].max()
    start = max(0, total_duration - 30)
    plot_eeg_data(data, start, total_duration)


class EEGChunkPlotter:
    """
    Interactive EEG chunk plotter with navigation and differential signal support.
    """
    
    def __init__(self, data):
        self.data = data
        self.current_chunk = 0
        self.chunks = []
        self.plot_type = 'raw'  # 'raw' or 'differential'
        self.differential_groups = {}
        self.fig = None
        self.axes = None
        
    def create_chunks(self, start_time, end_time, chunk_duration):
        """
        Create time chunks for plotting.
        
        Parameters:
        -----------
        start_time : float
            Start time in seconds
        end_time : float
            End time in seconds
        chunk_duration : float
            Duration of each chunk in seconds
        """
        self.chunks = []
        current_time = start_time
        
        while current_time + chunk_duration <= end_time:
            chunk_start = current_time
            chunk_end = current_time + chunk_duration
            self.chunks.append((chunk_start, chunk_end))
            current_time += chunk_duration
        
        print(f"Created {len(self.chunks)} chunks of {chunk_duration}s each")
        print(f"Time range: {start_time}s - {self.chunks[-1][1]}s")
        
    def plot_raw_signals(self, start_time, end_time):
        """Plot raw EEG signals for all channels."""
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        # Filter data for time range
        time_mask = (self.data['time'] >= start_time) & (self.data['time'] <= end_time)
        filtered_data = self.data[time_mask].copy()
        
        if len(filtered_data) == 0:
            print(f"No data found in range {start_time}-{end_time}s")
            return
        
        # Get pin columns
        pin_columns = [col for col in self.data.columns if col != 'time']
        
        # Plot each channel
        for i, pin_name in enumerate(pin_columns):
            if i < len(self.axes):
                self.axes[i].plot(filtered_data['time'], filtered_data[pin_name], 
                                linewidth=1, color=f'C{i}', alpha=0.8)
                self.axes[i].set_ylabel(f'{pin_name}\\n(µV)', fontsize=9)
                self.axes[i].grid(True, alpha=0.3)
                self.axes[i].set_title(f'{pin_name}', fontsize=10)
                
                # Optimize y-axis range
                y_mean = filtered_data[pin_name].mean()
                y_std = filtered_data[pin_name].std()
                self.axes[i].set_ylim(y_mean - 3*y_std, y_mean + 3*y_std)
        
        # Set x-axis label only on last subplot
        self.axes[-1].set_xlabel('Time (s)', fontsize=10)
        
    def plot_differential_signals(self, start_time, end_time):
        """Plot differential signals based on groups."""
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        # Filter data for time range
        time_mask = (self.data['time'] >= start_time) & (self.data['time'] <= end_time)
        filtered_data = self.data[time_mask].copy()
        
        if len(filtered_data) == 0:
            print(f"No data found in range {start_time}-{end_time}s")
            return
        
        ax_idx = 0
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        
        for group_name, pin_pairs in self.differential_groups.items():
            if ax_idx >= len(self.axes):
                break
                
            self.axes[ax_idx].set_title(f'Group: {group_name}', fontsize=11, fontweight='bold')
            
            for i, (pin1, pin2) in enumerate(pin_pairs):
                pin1_name = f'pin_{pin1}'
                pin2_name = f'pin_{pin2}'
                
                # Check if pins exist in data
                if pin1_name in filtered_data.columns and pin2_name in filtered_data.columns:
                    # Calculate differential signal
                    diff_signal = filtered_data[pin1_name] - filtered_data[pin2_name]
                    
                    color = colors[i % len(colors)]
                    self.axes[ax_idx].plot(filtered_data['time'], diff_signal, 
                                         linewidth=1, color=color, 
                                         label=f'{pin1_name} - {pin2_name}', alpha=0.8)
                else:
                    print(f"Warning: Pins {pin1_name} or {pin2_name} not found in data")
            
            self.axes[ax_idx].set_ylabel('Differential\\n(µV)', fontsize=9)
            self.axes[ax_idx].grid(True, alpha=0.3)
            self.axes[ax_idx].legend(fontsize=8, loc='upper right')
            
            ax_idx += 1
        
        # Hide unused subplots
        for i in range(ax_idx, len(self.axes)):
            self.axes[i].set_visible(False)
        
        # Set x-axis label
        if ax_idx > 0:
            self.axes[ax_idx-1].set_xlabel('Time (s)', fontsize=10)
    
    def update_plot(self):
        """Update the current plot."""
        if not self.chunks:
            print("No chunks available. Call create_chunks() first.")
            return
        
        if self.current_chunk >= len(self.chunks):
            self.current_chunk = len(self.chunks) - 1
        elif self.current_chunk < 0:
            self.current_chunk = 0
        
        start_time, end_time = self.chunks[self.current_chunk]
        
        if self.plot_type == 'raw':
            self.plot_raw_signals(start_time, end_time)
        else:
            self.plot_differential_signals(start_time, end_time)
        
        # Update main title
        self.fig.suptitle(f'EEG Chunk {self.current_chunk + 1}/{len(self.chunks)} ' +
                         f'({start_time:.1f}-{end_time:.1f}s) - {self.plot_type.upper()} MODE\\n' +
                         'Navigation: ← → keys or mouse clicks | Toggle: T key | Quit: Q key', 
                         fontsize=12, fontweight='bold')
        
        self.fig.canvas.draw()
        
    def on_key_press(self, event):
        """Handle key press events."""
        if event.key == 'right' or event.key == 'n':
            self.current_chunk = min(self.current_chunk + 1, len(self.chunks) - 1)
            self.update_plot()
        elif event.key == 'left' or event.key == 'p':
            self.current_chunk = max(self.current_chunk - 1, 0)
            self.update_plot()
        elif event.key == 't':
            # Toggle between raw and differential mode
            if self.differential_groups:
                self.plot_type = 'differential' if self.plot_type == 'raw' else 'raw'
                self.setup_subplots()
                self.update_plot()
            else:
                print("No differential groups defined. Use set_differential_groups() first.")
        elif event.key == 'q':
            plt.close(self.fig)
    
    def on_mouse_click(self, event):
        """Handle mouse click events."""
        if event.button == 1:  # Left click - next chunk
            self.current_chunk = min(self.current_chunk + 1, len(self.chunks) - 1)
            self.update_plot()
        elif event.button == 3:  # Right click - previous chunk
            self.current_chunk = max(self.current_chunk - 1, 0)
            self.update_plot()
    
    def setup_subplots(self):
        """Setup subplot layout based on plot type."""
        if self.plot_type == 'raw':
            pin_columns = [col for col in self.data.columns if col != 'time']
            n_subplots = len(pin_columns)
        else:
            n_subplots = len(self.differential_groups)
        
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig, self.axes = plt.subplots(n_subplots, 1, figsize=(15, max(8, n_subplots * 1.5)), 
                                          sharex=True)
        
        # Ensure axes is always a list
        if n_subplots == 1:
            self.axes = [self.axes]
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
    
    def set_differential_groups(self, groups):
        """
        Set differential signal groups.
        
        Parameters:
        -----------
        groups : dict
            Dictionary with group names as keys and list of (pin1, pin2) tuples as values
            Example: {'group1': [(4,1), (1,5), (5,6)], 'group2': [(4,2), (2,3)]}
        """
        self.differential_groups = groups
        print(f"Set {len(groups)} differential groups:")
        for group_name, pairs in groups.items():
            print(f"  {group_name}: {pairs}")
    
    def plot_chunks_raw(self, start_time, end_time, chunk_duration):
        """
        Plot raw EEG signals in chunks with navigation.
        
        Parameters:
        -----------
        start_time : float
            Start time in seconds
        end_time : float  
            End time in seconds
        chunk_duration : float
            Duration of each chunk in seconds
        """
        self.plot_type = 'raw'
        self.create_chunks(start_time, end_time, chunk_duration)
        self.setup_subplots()
        self.current_chunk = 0
        self.update_plot()
        plt.show()
    
    def plot_chunks_differential(self, start_time, end_time, chunk_duration, groups):
        """
        Plot differential EEG signals in chunks with navigation.
        
        Parameters:
        -----------
        start_time : float
            Start time in seconds
        end_time : float
            End time in seconds  
        chunk_duration : float
            Duration of each chunk in seconds
        groups : dict
            Differential signal groups
        """
        self.set_differential_groups(groups)
        self.plot_type = 'differential'
        self.create_chunks(start_time, end_time, chunk_duration)
        self.setup_subplots()
        self.current_chunk = 0
        self.update_plot()
        plt.show()

class EnhancedEEGChunkPlotter:
    """
    Enhanced Interactive EEG chunk plotter with advanced features.
    Includes spectrogram view, statistics, and improved navigation.
    """
    
    def __init__(self, data, sampling_rate=1000):
        self.data = data
        self.sampling_rate = sampling_rate
        self.current_chunk = 0
        self.chunks = []
        self.plot_mode = 'raw'  # 'raw', 'differential', 'spectrogram', 'statistics'
        self.differential_groups = {}
        self.fig = None
        self.axes = None
        self.info_text = None
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    def create_chunks(self, start_time, end_time, chunk_duration, overlap=0.0):
        """
        Create time chunks for plotting with optional overlap.
        
        Parameters:
        -----------
        start_time : float
            Start time in seconds
        end_time : float
            End time in seconds
        chunk_duration : float
            Duration of each chunk in seconds
        overlap : float
            Overlap between chunks as fraction (0-1)
        """
        self.chunks = []
        step_size = chunk_duration * (1 - overlap)
        current_time = start_time
        
        while current_time + chunk_duration <= end_time:
            chunk_start = current_time
            chunk_end = current_time + chunk_duration
            self.chunks.append((chunk_start, chunk_end))
            current_time += step_size
        
        print(f"Created {len(self.chunks)} chunks of {chunk_duration}s each")
        if overlap > 0:
            print(f"Overlap: {overlap*100:.1f}%")
        print(f"Time range: {start_time}s - {self.chunks[-1][1]:.1f}s")
        
    def get_chunk_data(self, chunk_idx):
        """Get data for a specific chunk."""
        if chunk_idx >= len(self.chunks):
            return None
        
        start_time, end_time = self.chunks[chunk_idx]
        time_mask = (self.data['time'] >= start_time) & (self.data['time'] <= end_time)
        return self.data[time_mask].copy()
    
    def calculate_statistics(self, chunk_data):
        """Calculate statistics for the chunk data."""
        pin_columns = [col for col in chunk_data.columns if col != 'time']
        stats = {}
        
        for pin in pin_columns:
            signal_data = chunk_data[pin].values
            stats[pin] = {
                'mean': np.mean(signal_data),
                'std': np.std(signal_data),
                'min': np.min(signal_data),
                'max': np.max(signal_data),
                'rms': np.sqrt(np.mean(signal_data**2)),
                'peak_to_peak': np.max(signal_data) - np.min(signal_data)
            }
        
        return stats
    
    def plot_raw_signals_enhanced(self, start_time, end_time):
        """Plot enhanced raw EEG signals with improved styling."""
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        chunk_data = self.get_chunk_data(self.current_chunk)
        if chunk_data is None or len(chunk_data) == 0:
            print(f"No data found in chunk {self.current_chunk}")
            return
        
        # Get pin columns
        pin_columns = [col for col in self.data.columns if col != 'time']
        
        # Plot each channel with enhanced styling
        for i, pin_name in enumerate(pin_columns):
            if i < len(self.axes):
                color = self.color_palette[i % len(self.color_palette)]
                
                # Plot signal
                self.axes[i].plot(chunk_data['time'], chunk_data[pin_name], 
                                linewidth=1.2, color=color, alpha=0.8)
                
                # Add zero line
                self.axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
                
                # Styling
                self.axes[i].set_ylabel(f'{pin_name}\\n(µV)', fontsize=9, fontweight='bold')
                self.axes[i].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                self.axes[i].set_facecolor('#fafafa')
                
                # Optimize y-axis range
                signal_data = chunk_data[pin_name]
                y_mean = signal_data.mean()
                y_std = signal_data.std()
                padding = max(y_std * 0.1, 1)  # Minimum padding of 1 µV
                self.axes[i].set_ylim(y_mean - 3*y_std - padding, y_mean + 3*y_std + padding)
                
                # Add statistics text
                stats_text = f'μ={y_mean:.1f} σ={y_std:.1f}'
                self.axes[i].text(0.98, 0.95, stats_text, transform=self.axes[i].transAxes,
                                ha='right', va='top', fontsize=8, 
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Set x-axis label only on last subplot
        self.axes[-1].set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    
    def plot_differential_signals(self, start_time, end_time):
        """Plot differential signals based on groups."""
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        chunk_data = self.get_chunk_data(self.current_chunk)
        if chunk_data is None or len(chunk_data) == 0:
            print(f"No data found in chunk {self.current_chunk}")
            return
        
        ax_idx = 0
        colors = self.color_palette
        
        for group_name, pin_pairs in self.differential_groups.items():
            if ax_idx >= len(self.axes):
                break
                
            self.axes[ax_idx].set_title(f'Group: {group_name}', fontsize=11, fontweight='bold')
            
            for i, (pin1, pin2) in enumerate(pin_pairs):
                pin1_name = f'pin_{pin1}'
                pin2_name = f'pin_{pin2}'
                
                # Check if pins exist in data
                if pin1_name in chunk_data.columns and pin2_name in chunk_data.columns:
                    # Calculate differential signal
                    diff_signal = chunk_data[pin1_name] - chunk_data[pin2_name]
                    
                    color = colors[i % len(colors)]
                    self.axes[ax_idx].plot(chunk_data['time'], diff_signal, 
                                         linewidth=1.2, color=color, 
                                         label=f'{pin1_name} - {pin2_name}', alpha=0.8)
                    
                    # Add zero line
                    self.axes[ax_idx].axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
                else:
                    print(f"Warning: Pins {pin1_name} or {pin2_name} not found in data")
            
            self.axes[ax_idx].set_ylabel('Differential\\n(µV)', fontsize=9, fontweight='bold')
            self.axes[ax_idx].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            self.axes[ax_idx].set_facecolor('#fafafa')
            self.axes[ax_idx].legend(fontsize=8, loc='upper right')
            
            ax_idx += 1
        
        # Hide unused subplots
        for i in range(ax_idx, len(self.axes)):
            self.axes[i].set_visible(False)
        
        # Set x-axis label
        if ax_idx > 0:
            self.axes[ax_idx-1].set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    
    def plot_spectrogram(self, start_time, end_time):
        """Plot spectrograms for selected channels."""
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        chunk_data = self.get_chunk_data(self.current_chunk)
        if chunk_data is None or len(chunk_data) == 0:
            print(f"No data found in chunk {self.current_chunk}")
            return
        
        # Get first few pin columns for spectrogram
        pin_columns = [col for col in self.data.columns if col != 'time']
        max_spectrograms = min(len(pin_columns), len(self.axes))
        
        for i in range(max_spectrograms):
            pin_name = pin_columns[i]
            signal_data = chunk_data[pin_name].values
            time_data = chunk_data['time'].values
            
            # Calculate spectrogram
            frequencies, times, Sxx = signal.spectrogram(
                signal_data, 
                fs=self.sampling_rate,
                window='hann',
                nperseg=min(256, len(signal_data)//4),
                noverlap=None
            )
            
            # Convert to dB
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            # Plot spectrogram
            im = self.axes[i].pcolormesh(
                times + time_data[0],  # Adjust time offset
                frequencies[:50],  # Show only up to 50 Hz
                Sxx_db[:50, :],
                shading='gouraud',
                cmap='viridis'
            )
            
            self.axes[i].set_ylabel(f'{pin_name}\\nFreq (Hz)', fontsize=9, fontweight='bold')
            self.axes[i].set_title(f'Spectrogram - {pin_name}', fontsize=10)
            
            # Add colorbar
            plt.colorbar(im, ax=self.axes[i], label='Power (dB)', shrink=0.8)
        
        # Hide unused subplots
        for i in range(max_spectrograms, len(self.axes)):
            self.axes[i].set_visible(False)
        
        # Set x-axis label
        if max_spectrograms > 0:
            self.axes[max_spectrograms-1].set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    
    def plot_statistics_view(self, start_time, end_time):
        """Plot statistical overview of all channels."""
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        chunk_data = self.get_chunk_data(self.current_chunk)
        if chunk_data is None or len(chunk_data) == 0:
            print(f"No data found in chunk {self.current_chunk}")
            return
        
        # Calculate statistics
        stats = self.calculate_statistics(chunk_data)
        pin_names = list(stats.keys())
        
        # Prepare data for plotting
        metrics = ['mean', 'std', 'rms', 'peak_to_peak']
        n_metrics = len(metrics)
        
        # Use first few axes for statistics plots
        for i, metric in enumerate(metrics):
            if i < len(self.axes):
                values = [stats[pin][metric] for pin in pin_names]
                colors = [self.color_palette[j % len(self.color_palette)] for j in range(len(pin_names))]
                
                bars = self.axes[i].bar(range(len(pin_names)), values, color=colors, alpha=0.7)
                self.axes[i].set_title(f'{metric.upper().replace("_", " ")}', fontsize=11, fontweight='bold')
                self.axes[i].set_ylabel('µV', fontsize=9)
                self.axes[i].set_xticks(range(len(pin_names)))
                self.axes[i].set_xticklabels(pin_names, rotation=45, ha='right')
                self.axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    self.axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        for i in range(n_metrics, len(self.axes)):
            self.axes[i].set_visible(False)
    
    def set_differential_groups(self, groups):
        """
        Set differential signal groups.
        
        Parameters:
        -----------
        groups : dict
            Dictionary with group names as keys and list of (pin1, pin2) tuples as values
            Example: {'group1': [(4,1), (1,5), (5,6)], 'group2': [(4,2), (2,3)]}
        """
        self.differential_groups = groups
        print(f"Set {len(groups)} differential groups:")
        for group_name, pairs in groups.items():
            print(f"  {group_name}: {pairs}")
    
    def update_plot(self):
        """Update the current plot based on mode."""
        if not self.chunks:
            print("No chunks available. Call create_chunks() first.")
            return
        
        if self.current_chunk >= len(self.chunks):
            self.current_chunk = len(self.chunks) - 1
        elif self.current_chunk < 0:
            self.current_chunk = 0
        
        start_time, end_time = self.chunks[self.current_chunk]
        
        # Plot based on current mode
        if self.plot_mode == 'raw':
            self.plot_raw_signals_enhanced(start_time, end_time)
        elif self.plot_mode == 'differential' and self.differential_groups:
            self.plot_differential_signals(start_time, end_time)
        elif self.plot_mode == 'spectrogram':
            self.plot_spectrogram(start_time, end_time)
        elif self.plot_mode == 'statistics':
            self.plot_statistics_view(start_time, end_time)
        
        # Update title and info
        mode_info = {
            'raw': 'RAW SIGNALS',
            'differential': 'DIFFERENTIAL SIGNALS', 
            'spectrogram': 'SPECTROGRAM VIEW',
            'statistics': 'STATISTICAL VIEW'
        }
        
        title = (f'EEG Analysis - Chunk {self.current_chunk + 1}/{len(self.chunks)} '
                f'({start_time:.1f}-{end_time:.1f}s) - {mode_info[self.plot_mode]}\\n'
                f'Navigation: ←→ (chunks) | ↑↓ (modes) | S (stats) | G (spectrogram) | Q (quit)')
        
        self.fig.suptitle(title, fontsize=12, fontweight='bold', y=0.95)
        
        # Update info text if available
        chunk_data = self.get_chunk_data(self.current_chunk)
        if chunk_data is not None and hasattr(self, 'info_text'):
            duration = end_time - start_time
            n_samples = len(chunk_data)
            sampling_info = f'Duration: {duration:.1f}s | Samples: {n_samples} | SR: {self.sampling_rate}Hz'
            
            if hasattr(self.fig, 'text_info'):
                self.fig.text_info.set_text(sampling_info)
            else:
                self.fig.text_info = self.fig.text(0.02, 0.02, sampling_info, 
                                                 fontsize=9, transform=self.fig.transFigure)
        
        self.fig.canvas.draw()
        
    def on_key_press(self, event):
        """Handle enhanced key press events."""
        if event.key == 'right' or event.key == 'n':
            self.current_chunk = min(self.current_chunk + 1, len(self.chunks) - 1)
            self.update_plot()
        elif event.key == 'left' or event.key == 'p':
            self.current_chunk = max(self.current_chunk - 1, 0)
            self.update_plot()
        elif event.key == 'up':
            # Cycle through modes
            modes = ['raw', 'differential'] if self.differential_groups else ['raw']
            modes.extend(['spectrogram', 'statistics'])
            current_idx = modes.index(self.plot_mode)
            self.plot_mode = modes[(current_idx + 1) % len(modes)]
            self.setup_subplots()
            self.update_plot()
        elif event.key == 'down':
            # Cycle through modes (reverse)
            modes = ['raw', 'differential'] if self.differential_groups else ['raw']
            modes.extend(['spectrogram', 'statistics'])
            current_idx = modes.index(self.plot_mode)
            self.plot_mode = modes[(current_idx - 1) % len(modes)]
            self.setup_subplots()
            self.update_plot()
        elif event.key == 's':
            # Switch to statistics mode
            self.plot_mode = 'statistics'
            self.setup_subplots()
            self.update_plot()
        elif event.key == 'g':
            # Switch to spectrogram mode
            self.plot_mode = 'spectrogram'
            self.setup_subplots()
            self.update_plot()
        elif event.key == 'r':
            # Switch to raw mode
            self.plot_mode = 'raw'
            self.setup_subplots()
            self.update_plot()
        elif event.key == 'd' and self.differential_groups:
            # Switch to differential mode
            self.plot_mode = 'differential'
            self.setup_subplots()
            self.update_plot()
        elif event.key == 'q':
            plt.close(self.fig)
        elif event.key == 'h':
            # Show help
            self.show_help()
    
    def on_mouse_click(self, event):
        """Handle mouse click events."""
        if event.button == 1:  # Left click - next chunk
            self.current_chunk = min(self.current_chunk + 1, len(self.chunks) - 1)
            self.update_plot()
        elif event.button == 3:  # Right click - previous chunk
            self.current_chunk = max(self.current_chunk - 1, 0)
            self.update_plot()
    
    def show_help(self):
        """Display help information."""
        help_text = """
        🎯 Enhanced EEG Chunk Plotter - Keyboard Shortcuts:
        
        Navigation:
        ← / →     : Previous/Next chunk
        ↑ / ↓     : Cycle through view modes
        
        View Modes:
        R         : Raw signals view
        D         : Differential signals view (if groups defined)
        G         : Spectrogram view
        S         : Statistics view
        
        Other:
        H         : Show this help
        Q         : Quit
        
        Mouse:
        Left click  : Next chunk
        Right click : Previous chunk
        """
        print(help_text)
    
    def setup_subplots(self):
        """Setup subplot layout based on plot mode."""
        if self.plot_mode == 'raw':
            pin_columns = [col for col in self.data.columns if col != 'time']
            n_subplots = len(pin_columns)
            figsize = (15, max(10, n_subplots * 1.2))
        elif self.plot_mode == 'differential':
            n_subplots = len(self.differential_groups)
            figsize = (15, max(8, n_subplots * 2))
        elif self.plot_mode == 'spectrogram':
            pin_columns = [col for col in self.data.columns if col != 'time']
            n_subplots = min(4, len(pin_columns))  # Max 4 spectrograms
            figsize = (15, max(10, n_subplots * 2.5))
        elif self.plot_mode == 'statistics':
            n_subplots = 4  # For different statistical metrics
            figsize = (15, 10)
        
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig, self.axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
        
        # Ensure axes is always a list
        if n_subplots == 1:
            self.axes = [self.axes]
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.08)
    
    def plot_chunks_interactive(self, start_time, end_time, chunk_duration, overlap=0.0, mode='raw'):
        """
        Start interactive chunk plotting with enhanced features.
        
        Parameters:
        -----------
        start_time : float
            Start time in seconds
        end_time : float  
            End time in seconds
        chunk_duration : float
            Duration of each chunk in seconds
        overlap : float
            Overlap between chunks (0-1)
        mode : str
            Initial plot mode ('raw', 'differential', 'spectrogram', 'statistics')
        """
        self.plot_mode = mode
        self.create_chunks(start_time, end_time, chunk_duration, overlap)
        self.setup_subplots()
        self.current_chunk = 0
        self.update_plot()
        self.show_help()
        plt.show()

def plot_differential_scaled_7uv_per_mm(data, start_time, end_time, groups, 
                                        figsize_inches=(15, 10), dpi=100,
                                        apply_filter=False, lowcut=0.5, highcut=70.0, 
                                        filter_type='butterworth', sampling_rate=1000):
    """
    Plot differential EEG signals with precise scaling: 7µV = 1mm on screen.
    Each group gets its own subplot with differential pairs spaced 1cm apart within the subplot.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        EEG 데이터 (time 컬럼과 pin 컬럼들 포함)
    start_time : float
        시작 시간 (초)
    end_time : float  
        종료 시간 (초)
    groups : dict
        Dictionary with group names as keys and list of (pin1, pin2) tuples as values
        Example: {'group1': [(4,1), (1,5)], 'group2': [(4,2), (2,3)]}
    figsize_inches : tuple
        Figure size in inches (width, height)
    dpi : int
        Dots per inch for the figure
    apply_filter : bool
        Whether to apply bandpass filter before plotting (default: False)
    lowcut : float
        Low cutoff frequency for bandpass filter in Hz (default: 0.5)
    highcut : float
        High cutoff frequency for bandpass filter in Hz (default: 70.0)
    filter_type : str
        Filter type: 'butterworth', 'elliptic', 'chebyshev1', 'chebyshev2' (default: 'butterworth')
    sampling_rate : float
        Sampling rate in Hz (default: 1000)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Apply bandpass filter if requested
    if apply_filter:
        print(f"🔧 Applying {filter_type} bandpass filter: {lowcut}-{highcut} Hz")
        filtered_data = apply_bandpass_filter(data, lowcut, highcut, 
                                            fs=sampling_rate, filter_type=filter_type)
    else:
        filtered_data = data
    
    # Calculate mm to µV conversion
    # 1mm on screen = 7µV
    uv_per_mm = 7.0
    
    # Calculate vertical spacing for differential pairs within each group (1cm = 10mm)
    pair_spacing_mm = 10.0  # 1cm in mm
    pair_spacing_uv = pair_spacing_mm * uv_per_mm  # Convert to µV units
    
    # Filter data for time range
    time_mask = (filtered_data['time'] >= start_time) & (filtered_data['time'] <= end_time)
    plot_data = filtered_data[time_mask].copy()
    
    if len(plot_data) == 0:
        print(f"No data found in time range {start_time}-{end_time} seconds")
        return
    
    # Calculate subplot layout
    n_groups = len(groups)
    
    # Adjust figure height based on number of groups
    subplot_height = figsize_inches[1] / n_groups if n_groups > 1 else figsize_inches[1]
    
    # Create subplots - one for each group
    fig, axes = plt.subplots(n_groups, 1, figsize=figsize_inches, dpi=dpi, sharex=True)
    
    # Ensure axes is always a list
    if n_groups == 1:
        axes = [axes]
    
    # Color palette for different signals within groups
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Process each group in its own subplot
    for group_idx, (group_name, pin_pairs) in enumerate(groups.items()):
        ax = axes[group_idx]
        
        # Track y-axis limits for this subplot
        y_min_subplot = float('inf')
        y_max_subplot = float('-inf')
        
        # Calculate vertical positions for differential pairs within this group
        n_pairs = len(pin_pairs)
        center_offset = (n_pairs - 1) * pair_spacing_uv / 2
        
        print(f"Plotting group '{group_name}' with {n_pairs} differential pairs")
        
        # Plot each differential pair in the group
        for pair_idx, (pin1, pin2) in enumerate(pin_pairs):
            pin1_name = f'pin_{pin1}'
            pin2_name = f'pin_{pin2}'
            
            # Check if pins exist in data
            if pin1_name in plot_data.columns and pin2_name in plot_data.columns:
                # Calculate differential signal
                diff_signal = plot_data[pin1_name] - plot_data[pin2_name]
                
                # Calculate vertical offset for this pair (1cm spacing)
                pair_offset = pair_idx * pair_spacing_uv - center_offset
                offset_signal = diff_signal + pair_offset
                
                # Select color
                color = colors[pair_idx % len(colors)]
                
                # Plot the signal
                ax.plot(plot_data['time'], offset_signal, 
                       linewidth=1.2, color=color, 
                       label=f'{pin1_name}-{pin2_name}', 
                       alpha=0.8)
                
                # Update subplot y limits
                y_min_subplot = min(y_min_subplot, offset_signal.min())
                y_max_subplot = max(y_max_subplot, offset_signal.max())
                
                # Add zero line for this differential pair
                ax.axhline(y=pair_offset, color='gray', linestyle='--', 
                          alpha=0.4, linewidth=0.8)
                
            else:
                print(f"Warning: Pins {pin1_name} or {pin2_name} not found in data")
        
        # Configure this subplot
        ax.set_ylabel(f'{group_name}\nDifferential (7µV = 1mm)', fontsize=10, fontweight='bold')
        ax.set_title(f'Group: {group_name}', fontsize=11, fontweight='bold')
        
        # Ensure time axis fills the graph completely
        ax.set_xlim(start_time, end_time)
        
        # Set y-axis limits with padding
        if y_min_subplot != float('inf') and y_max_subplot != float('-inf'):
            y_range = y_max_subplot - y_min_subplot
            padding = max(y_range * 0.05, uv_per_mm)  # At least 1mm padding
            ax.set_ylim(y_min_subplot - padding, y_max_subplot + padding)
            
            # Create custom y-axis ticks based on 7µV = 1mm scale
            tick_spacing_uv = uv_per_mm  # 7µV per tick
            
            # Find appropriate tick range for this subplot
            y_tick_start = np.floor((y_min_subplot - padding) / tick_spacing_uv) * tick_spacing_uv
            y_tick_end = np.ceil((y_max_subplot + padding) / tick_spacing_uv) * tick_spacing_uv
            
            y_ticks = np.arange(y_tick_start, y_tick_end + tick_spacing_uv, tick_spacing_uv)
            
            ax.set_yticks(y_ticks)
            
            # Create custom y-tick labels showing both µV and mm
            y_tick_labels = []
            for tick in y_ticks:
                mm_value = tick / uv_per_mm
                y_tick_labels.append(f'{tick:.0f}µV\\n({mm_value:.1f}mm)')
            
            ax.set_yticklabels(y_tick_labels, fontsize=8)
        
        # Add grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#fafafa')
        
        # Add legend for this subplot
        ax.legend(loc='upper right', fontsize=8)
    
    # Set x-axis label only on the bottom subplot
    axes[-1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    
    # Add overall title
    filter_info = f" | Filtered: {lowcut}-{highcut}Hz ({filter_type})" if apply_filter else ""
    title = (f'Differential EEG Signals by Groups ({start_time:.1f}-{end_time:.1f}s){filter_info}\\n'
            f'Scale: 7µV = 1mm | Differential pairs spaced 1cm apart within each group')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Add scaling information box on the first subplot
    filter_text = f'🔧 Filter: {lowcut}-{highcut}Hz ({filter_type})\\n' if apply_filter else ''
    info_text = (f'📏 Scaling: 7µV = 1mm\\n'
                f'{filter_text}'
                f'📊 Duration: {end_time-start_time:.1f}s\\n'
                f'🔗 Groups: {len(groups)}\\n'
                f'📈 Pairs: {sum(len(pairs) for pairs in groups.values())}')
    
    axes[0].text(0.02, 0.98, info_text, transform=axes[0].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Show the plot
    plt.show()
    
    # Print summary information
    filter_summary = f" with {filter_type} filter ({lowcut}-{highcut}Hz)" if apply_filter else ""
    print(f"\\n📊 Plot Summary{filter_summary}:")
    print(f"   Time range: {start_time:.1f} - {end_time:.1f} seconds ({end_time-start_time:.1f}s duration)")
    print(f"   Groups plotted: {len(groups)} (each in separate subplot)")
    print(f"   Total differential pairs: {sum(len(pairs) for pairs in groups.values())}")
    print(f"   Scaling: 7µV = 1mm")
    if apply_filter:
        print(f"   Filter applied: {filter_type} bandpass {lowcut}-{highcut}Hz")
    print(f"   Differential pair spacing within groups: 1cm ({pair_spacing_uv}µV)")
    for group_name, pairs in groups.items():
        print(f"   • {group_name}: {len(pairs)} differential pairs")


class DifferentialPlotter7uvPerMm:
    """
    Specialized plotter for differential signals with 7µV = 1mm scaling.
    Each group gets its own subplot with differential pairs spaced 1cm apart.
    Includes bandpass filtering capabilities.
    """
    
    def __init__(self, data, figsize_inches=(15, 10), dpi=100, sampling_rate=1000):
        self.data = data
        self.figsize_inches = figsize_inches
        self.dpi = dpi
        self.sampling_rate = sampling_rate
        self.uv_per_mm = 7.0
        self.pair_spacing_mm = 10.0  # 1cm
        self.pair_spacing_uv = self.pair_spacing_mm * self.uv_per_mm
        
        # Filter settings
        self.filter_applied = True
        self.filter_params = {}
        
        # Color palette
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def apply_filter(self, lowcut, highcut, filter_type='butterworth', order=4):
        """
        Apply bandpass filter to the data.
        
        Parameters:
        -----------
        lowcut : float
            Low cutoff frequency in Hz
        highcut : float
            High cutoff frequency in Hz
        filter_type : str
            Filter type: 'butterworth', 'elliptic', 'chebyshev1', 'chebyshev2'
        order : int
            Filter order
        """
        print(f"🔧 Applying {filter_type} bandpass filter: {lowcut}-{highcut} Hz")
        self.data = apply_bandpass_filter(self.data, lowcut, highcut, 
                                        fs=self.sampling_rate, 
                                        filter_type=filter_type, order=order)
        self.filter_applied = True
        self.filter_params = {
            'lowcut': lowcut,
            'highcut': highcut,
            'filter_type': filter_type,
            'order': order
        }
        print(f"✅ Filter applied successfully")
    
    def reset_filter(self, original_data):
        """
        Reset to original unfiltered data.
        
        Parameters:
        -----------
        original_data : pandas.DataFrame
            Original unfiltered data
        """
        self.data = original_data.copy()
        self.filter_applied = False
        self.filter_params = {}
        print("🔄 Reset to original unfiltered data")
    
    def plot_time_range(self, start_time, end_time, groups, title_suffix=""):
        """
        Plot differential signals for a specific time range with each group in its own subplot.
        
        Parameters:
        -----------
        start_time : float
            Start time in seconds
        end_time : float
            End time in seconds  
        groups : dict
            Dictionary with group names as keys and list of (pin1, pin2) tuples as values
        title_suffix : str
            Additional text to add to the title
        """
        # Filter data for time range
        time_mask = (self.data['time'] >= start_time) & (self.data['time'] <= end_time)
        filtered_data = self.data[time_mask].copy()
        
        if len(filtered_data) == 0:
            print(f"No data found in time range {start_time}-{end_time} seconds")
            return None, None
        
        # Calculate subplot layout
        n_groups = len(groups)
        
        # Create subplots - one for each group
        fig, axes = plt.subplots(n_groups, 1, figsize=self.figsize_inches, 
                               dpi=self.dpi, sharex=True)
        
        # Ensure axes is always a list
        if n_groups == 1:
            axes = [axes]
        
        # Process each group in its own subplot
        for group_idx, (group_name, pin_pairs) in enumerate(groups.items()):
            ax = axes[group_idx]
            
            # Track y-axis limits for this subplot
            y_min_subplot = float('inf')
            y_max_subplot = float('-inf')
            
            # Calculate vertical positions for differential pairs within this group
            n_pairs = len(pin_pairs)
            center_offset = (n_pairs - 1) * self.pair_spacing_uv / 2
            
            # Plot each differential pair in the group
            for pair_idx, (pin1, pin2) in enumerate(pin_pairs):
                pin1_name = f'pin_{pin1}'
                pin2_name = f'pin_{pin2}'
                
                # Check if pins exist in data
                if pin1_name in filtered_data.columns and pin2_name in filtered_data.columns:
                    # Calculate differential signal
                    diff_signal = filtered_data[pin1_name] - filtered_data[pin2_name]
                    
                    # Calculate vertical offset for this pair (1cm spacing)
                    pair_offset = pair_idx * self.pair_spacing_uv - center_offset
                    offset_signal = diff_signal + pair_offset
                    
                    # Select color
                    color = self.colors[pair_idx % len(self.colors)]
                    
                    # Plot the signal
                    ax.plot(filtered_data['time'], offset_signal, 
                           linewidth=1.2, color=color, 
                           label=f'{pin1_name}-{pin2_name}', 
                           alpha=0.8)
                    
                    # Update subplot y limits
                    y_min_subplot = min(y_min_subplot, offset_signal.min())
                    y_max_subplot = max(y_max_subplot, offset_signal.max())
                    
                    # Add zero line for this differential pair
                    ax.axhline(y=pair_offset, color='gray', linestyle='--', 
                              alpha=0.4, linewidth=0.8)
                
                else:
                    print(f"Warning: Pins {pin1_name} or {pin2_name} not found in data")
            
            # Configure this subplot
            self._configure_subplot(ax, group_name, start_time, end_time, 
                                  y_min_subplot, y_max_subplot)
        
        # Set x-axis label only on the bottom subplot
        axes[-1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        
        # Add overall title
        filter_info = ""
        if self.filter_applied:
            params = self.filter_params
            filter_info = f" | Filtered: {params['lowcut']}-{params['highcut']}Hz ({params['filter_type']})"
        
        title = (f'Differential EEG Signals by Groups ({start_time:.1f}-{end_time:.1f}s){filter_info}\\n'
                f'Scale: 7µV = 1mm | Differential pairs spaced 1cm apart {title_suffix}')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Add info box on first subplot
        self._add_info_box(axes[0], start_time, end_time, groups)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        return fig, axes
    
    def _configure_subplot(self, ax, group_name, start_time, end_time, y_min, y_max):
        """Configure individual subplot properties."""
        ax.set_ylabel(f'{group_name}\\nDifferential (7µV = 1mm)', fontsize=10, fontweight='bold')
        ax.set_title(f'Group: {group_name}', fontsize=11, fontweight='bold')
        
        # Set axis limits
        ax.set_xlim(start_time, end_time)
        
        if y_min != float('inf') and y_max != float('-inf'):
            y_range = y_max - y_min
            padding = max(y_range * 0.05, self.uv_per_mm)  # At least 1mm padding
            ax.set_ylim(y_min - padding, y_max + padding)
            
            # Create custom y-axis ticks (every 1mm = 7µV)
            tick_spacing_uv = self.uv_per_mm
            y_tick_start = np.floor((y_min - padding) / tick_spacing_uv) * tick_spacing_uv
            y_tick_end = np.ceil((y_max + padding) / tick_spacing_uv) * tick_spacing_uv
            
            y_ticks = np.arange(y_tick_start, y_tick_end + tick_spacing_uv, tick_spacing_uv)
            ax.set_yticks(y_ticks)
            
            # Create tick labels with both µV and mm
            y_tick_labels = []
            for tick in y_ticks:
                mm_value = tick / self.uv_per_mm
                y_tick_labels.append(f'{tick:.0f}µV\\n({mm_value:.1f}mm)')
            
            ax.set_yticklabels(y_tick_labels, fontsize=8)
        
        # Add grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#fafafa')
        ax.legend(loc='upper right', fontsize=8)
    
    def _add_info_box(self, ax, start_time, end_time, groups):
        """Add information box to the plot."""
        duration = end_time - start_time
        total_pairs = sum(len(pairs) for pairs in groups.values())
        
        info_text = (f'📏 Scaling: 7µV = 1mm\\n'
                    f'{filter_text}'
                    f'📊 Duration: {duration:.1f}s\\n'
                    f'🔗 Groups: {len(groups)}\\n'
                    f'📈 Pairs: {total_pairs}')
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    def plot_multiple_chunks(self, start_time, end_time, chunk_duration, groups, overlap=0.0):
        """
        Plot multiple chunks with consistent scaling, each group in separate subplots.
        
        Parameters:
        -----------
        start_time : float
            Start time in seconds
        end_time : float
            End time in seconds
        chunk_duration : float
            Duration of each chunk in seconds
        groups : dict
            Differential signal groups
        overlap : float
            Overlap between chunks (0-1)
        """
        # Create chunks
        chunks = []
        step_size = chunk_duration * (1 - overlap)
        current_time = start_time
        
        while current_time + chunk_duration <= end_time:
            chunks.append((current_time, current_time + chunk_duration))
            current_time += step_size
        
        print(f"Created {len(chunks)} chunks of {chunk_duration}s each")
        print(f"Each group will be displayed in its own subplot")
        
        # Plot each chunk
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            title_suffix = f"| Chunk {i+1}/{len(chunks)}"
            fig, axes = self.plot_time_range(chunk_start, chunk_end, groups, title_suffix)
            if fig is not None:
                plt.show()
                
                # Print chunk info
                print(f"\\nChunk {i+1}: {chunk_start:.1f}-{chunk_end:.1f}s")
                print(f"Groups displayed: {list(groups.keys())}")

def apply_bandpass_filter(data, lowcut, highcut, fs=1000, order=4, filter_type='butterworth'):
    """
    Apply bandpass filter to EEG data.
    
    Parameters:
    -----------
    data : pandas.DataFrame or numpy.array
        EEG data (if DataFrame, should have time column and pin columns)
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz (default: 1000)
    order : int
        Filter order (default: 4)
    filter_type : str
        Filter type: 'butterworth', 'elliptic', 'chebyshev1', 'chebyshev2' (default: 'butterworth')
    
    Returns:
    --------
    filtered_data : pandas.DataFrame or numpy.array
        Filtered data with same structure as input
    """
    from scipy.signal import butter, ellip, cheby1, cheby2, filtfilt
    import pandas as pd
    import numpy as np
    
    # Validate frequency parameters
    nyquist = fs / 2.0
    if lowcut >= nyquist or highcut >= nyquist:
        raise ValueError(f"Cutoff frequencies must be less than Nyquist frequency ({nyquist} Hz)")
    if lowcut >= highcut:
        raise ValueError("Low cutoff frequency must be less than high cutoff frequency")
    
    # Normalize frequencies
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design filter based on type
    if filter_type.lower() == 'butterworth':
        b, a = butter(order, [low, high], btype='band')
    elif filter_type.lower() == 'elliptic':
        b, a = ellip(order, 1, 40, [low, high], btype='band')
    elif filter_type.lower() == 'chebyshev1':
        b, a = cheby1(order, 1, [low, high], btype='band')
    elif filter_type.lower() == 'chebyshev2':
        b, a = cheby2(order, 40, [low, high], btype='band')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply filter
    if isinstance(data, pd.DataFrame):
        # DataFrame input - filter pin columns
        filtered_data = data.copy()
        pin_columns = [col for col in data.columns if col.startswith('pin_')]
        
        print(f"🔧 Applying {filter_type} bandpass filter ({lowcut}-{highcut} Hz) to {len(pin_columns)} channels...")
        
        for pin_col in pin_columns:
            # Apply zero-phase filtering to avoid phase distortion
            filtered_data[pin_col] = filtfilt(b, a, data[pin_col].values)
        
        print(f"✅ Filtering completed successfully")
        return filtered_data
    
    else:
        # Array input
        if data.ndim == 1:
            return filtfilt(b, a, data)
        else:
            # Multi-channel array
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered_data[:, i] = filtfilt(b, a, data[:, i])
            return filtered_data

def get_eeg_frequency_bands():
    """
    Get standard EEG frequency bands for easy reference.
    
    Returns:
    --------
    dict : Dictionary of frequency bands with their ranges
    """
    return {
        'delta': (0.5, 4.0),     # Deep sleep
        'theta': (4.0, 8.0),     # Drowsiness, meditation
        'alpha': (8.0, 13.0),    # Relaxed, eyes closed
        'beta': (13.0, 30.0),    # Alert, active thinking
        'gamma': (30.0, 100.0),  # High-level cognitive processing
        'broadband': (0.5, 50.0), # General EEG analysis
        'clinical': (0.5, 70.0),  # Clinical EEG
        'artifacts': (50.0, 200.0) # High-frequency artifacts
    }

def print_filter_info():
    """Print information about available filters and frequency bands."""
    bands = get_eeg_frequency_bands()
    
    print("🧠 EEG Frequency Bands:")
    for band_name, (low, high) in bands.items():
        print(f"   • {band_name.capitalize()}: {low}-{high} Hz")
    
    print(f"\n🔧 Available Filter Types:")
    print(f"   • butterworth: Maximally flat passband")
    print(f"   • elliptic: Steep transition, some ripple")
    print(f"   • chebyshev1: Ripple in passband")
    print(f"   • chebyshev2: Ripple in stopband")
    
    print(f"\n💡 Usage Examples:")
    print(f"   # Filter for alpha waves (8-13 Hz)")
    print(f"   filtered_data = apply_bandpass_filter(data, 8, 13)")
    print(f"   ")
    print(f"   # Clinical EEG range")
    print(f"   filtered_data = apply_bandpass_filter(data, 0.5, 70)")