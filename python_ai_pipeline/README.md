# Motor Anomaly Detection - AI Pipeline

This directory contains the Python pipeline for collecting, preprocessing, and training AI models for motor anomaly detection using data from the STM32H745 board.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Connect STM32 Board

1. Connect your STM32H745 board via USB
2. Identify the COM port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)
3. Update the `SERIAL_PORT` variable in the scripts

## Usage

### Step 1: Data Collection

```bash
python data_collector.py
```

This script will:
- Connect to your STM32H745 board
- Guide you through collecting vibration data for different motor conditions
- Save data in HDF5 format for efficient storage

**Collection Process:**
1. **Normal Motor**: Collect baseline data from healthy motor
2. **Imbalance**: Add weight to create imbalance, then collect data
3. **Bearing Fault**: Use motor with bearing wear or simulate bearing issues
4. **Misalignment**: Misalign motor shaft and collect data

### Step 2: Data Preprocessing

```bash
python data_preprocessor.py
```

This script will:
- Load collected HDF5 data
- Apply digital filters to remove noise
- Extract comprehensive features (time, frequency, statistical domains)
- Create visualizations of feature distributions
- Save processed features as CSV

### Step 3: Model Training (Coming Next)

```bash
python model_trainer.py  # Will be created in next step
```

## Data Collection Commands

Your STM32 board responds to these USB commands:

- `START_NORMAL` - Start collecting normal motor data
- `START_IMBALANCE` - Start collecting imbalance data  
- `START_BEARING` - Start collecting bearing fault data
- `START_MISALIGN` - Start collecting misalignment data
- `STOP` - Stop current data collection
- `GET_DATA` - Retrieve collected data
- `STATUS` - Get collection status
- `RESET` - Reset collection system

## File Structure

```
python_ai_pipeline/
├── data_collector.py      # STM32 communication and data collection
├── data_preprocessor.py   # Feature extraction and preprocessing
├── model_trainer.py       # AI model training (coming next)
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── collected_data/       # Raw data from STM32 (HDF5 files)
├── processed_data/       # Processed features (CSV files)
└── models/              # Trained AI models
```

## Features Extracted

### Time Domain Features (per axis)
- Mean, Standard Deviation, Variance
- RMS (Root Mean Square)
- Peak, Peak-to-Peak
- Crest Factor, Form Factor
- Skewness, Kurtosis
- Energy, Power

### Frequency Domain Features (per axis)
- Spectral Centroid
- Spectral Rolloff
- Spectral Flux
- Frequency Band Energy Ratios (Low/Mid/High)
- Dominant Frequency and Magnitude

### Statistical Features (cross-axis)
- Correlation between axes (XY, XZ, YZ)
- Magnitude statistics (mean, std, max, min)
- Energy ratios between axes

## Data Format

### Raw Data (HDF5)
```
sample_1/
├── x_data          # X-axis acceleration data
├── y_data          # Y-axis acceleration data
├── z_data          # Z-axis acceleration data
└── attributes/
    ├── fault_type  # 0=Normal, 1=Imbalance, 2=Bearing, 3=Misalign
    ├── sample_rate # Sampling rate (1000 Hz)
    └── duration_ms # Sample duration in milliseconds
```

### Processed Data (CSV)
Each row represents one sample with extracted features:
```
sample_id, fault_type, x_rms, y_rms, z_rms, x_peak, y_peak, z_peak, ...
```

## Troubleshooting

### STM32 Connection Issues
1. Check COM port in Device Manager (Windows) or `ls /dev/tty*` (Linux)
2. Ensure STM32 firmware is flashed and running
3. Try different baud rates (115200, 9600)
4. Check USB cable quality

### Data Collection Issues
1. Verify motor is properly mounted and running
2. Check accelerometer connection (I2C)
3. Monitor STM32 serial output for errors
4. Ensure adequate power supply

### Memory Issues
1. Use HDF5 format for large datasets (more efficient than CSV)
2. Process data in chunks if memory is limited
3. Consider reducing sample duration or rate for testing

## Next Steps

After collecting and preprocessing data:

1. **Model Training**: Train CNN/LSTM models for anomaly detection
2. **Model Optimization**: Quantize models for STM32 deployment
3. **Integration**: Deploy trained model back to STM32H745
4. **Validation**: Test real-time anomaly detection performance

## Support

For issues or questions:
1. Check STM32 serial output for error messages
2. Verify all connections and power supply
3. Review Python error logs for debugging information
