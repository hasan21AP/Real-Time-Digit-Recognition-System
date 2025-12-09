from ..constants import global_variables as glv
import psutil
import warnings
import time


warnings.filterwarnings("ignore", category=FutureWarning)

# Import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - CPU/RAM statistics will be limited")


########################################
#           PERFORMANCE MONITORING FUNCTIONS
########################################
def get_cpu_usage():
    """Get current CPU usage percentage"""
    if PSUTIL_AVAILABLE:
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            print(f"⚠️ CPU monitoring error: {e}")
            return 0
    return 0

def get_ram_usage():
    """Get current RAM usage in GB and percentage"""
    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            ram_used_gb = memory.used / (1024**3)  # Convert to GB
            ram_total_gb = memory.total / (1024**3)  # Convert to GB
            ram_percent = memory.percent
            return ram_used_gb, ram_total_gb, ram_percent
        except Exception as e:
            print(f"⚠️ RAM monitoring error: {e}")
            return 0, 0, 0
    return 0, 0, 0

def get_process_ram_usage():
    """Get RAM usage of current process in MB"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024**2)  # Convert to MB
        except Exception as e:
            print(f"⚠️ Process RAM monitoring error: {e}")
            return 0
    return 0

def print_exit_statistics():
    """Print comprehensive statistics when exiting the program"""
    print("\n" + "="*60)
    print("📊 PROGRAM EXIT STATISTICS")
    print("="*60)
    
    # Calculate runtime
    end_time = time.time()
    total_runtime = end_time - glv.start_time
    minutes = int(total_runtime // 60)
    seconds = int(total_runtime % 60)
    
    print(f"⏱️  Total Runtime: {minutes}m {seconds}s")
    print(f"📈 Total Frames Processed: {glv.total_frames_processed}")
    
    # Calculate average FPS
    if total_runtime > 0:
        avg_fps = glv.total_frames_processed / total_runtime
        print(f"🔄 Average FPS: {avg_fps:.2f}")
    
    # CPU Statistics
    cpu_usage = get_cpu_usage()
    print(f"🔧 CPU Usage: {cpu_usage:.1f}%")
    
    # RAM Statistics - System RAM
    if PSUTIL_AVAILABLE:
        ram_used, ram_total, ram_percent = get_ram_usage()
        print(f"🧠 System RAM Usage: {ram_used:.1f}/{ram_total:.1f} GB ({ram_percent:.1f}%)")
        
        # Process-specific RAM usage
        process_ram = get_process_ram_usage()
        print(f"🔍 Process RAM Usage: {process_ram:.1f} MB")
    else:
        print("🧠 RAM Stats: Install 'psutil' for detailed RAM monitoring")
    
    # Performance Statistics
    print(f"📸 Digits Detected: {glv.counter}")
    print(f"🔢 Unique Digits Tracked: {len(glv.processed_digits)}")
    
    # Efficiency metrics
    if glv.total_frames_processed > 0:
        frames_per_second = glv.total_frames_processed / total_runtime
        print(f"⚡ Overall Performance: {frames_per_second:.1f} FPS")
        
        if PSUTIL_AVAILABLE:
            memory_per_frame = get_process_ram_usage() / glv.total_frames_processed
            print(f"💪 Memory Efficiency: {memory_per_frame:.1f} MB per frame")
    
    print("="*60)
    print("✅ Thank you for using Car Racing Digit Recognition!")
    print("="*60)