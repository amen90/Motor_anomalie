#!/usr/bin/env python3
"""
Motor Anomaly Detection - Data Collection Module
This module handles communication with the STM32H745 board to collect
vibration data for AI model training.
"""

import serial
import time
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MotorFaultType(Enum):
    """Motor fault types matching STM32 firmware"""
    NORMAL = 0
    IMBALANCE = 1
    BEARING_FAULT = 2
    MISALIGNMENT = 3

@dataclass
class MotorSample:
    """Data structure for a motor vibration sample"""
    sample_id: int
    timestamp: int
    fault_type: MotorFaultType
    sample_rate: int
    duration_ms: int
    num_samples: int
    x_data: List[int]
    y_data: List[int]
    z_data: List[int]
    metadata: Dict

class STM32DataCollector:
    """
    Class to handle communication with STM32H745 board for data collection
    """
    
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 10.0):
        """
        Initialize the data collector
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Serial communication speed
            timeout: Communication timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.is_connected = False
        self.collected_samples = []
        
    def connect(self) -> bool:
        """
        Establish connection with STM32 board
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Clear any existing data
            self.serial_conn.flushInput()
            self.serial_conn.flushOutput()
            
            # Test connection
            self.send_command("STATUS")
            response = self.read_response()
            
            if response and "STATUS" in response:
                self.is_connected = True
                logger.info(f"Connected to STM32 on port {self.port}")
                return True
            else:
                logger.error("Failed to establish communication with STM32")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False
    
    def disconnect(self):
        """Close connection with STM32 board"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            logger.info("Disconnected from STM32")
    
    def send_command(self, command: str) -> bool:
        """
        Send command to STM32
        
        Args:
            command: Command string to send
            
        Returns:
            bool: True if command sent successfully
        """
        if not self.is_connected or not self.serial_conn:
            logger.error("Not connected to STM32")
            return False
        
        try:
            command_bytes = f"{command}\r\n".encode('utf-8')
            self.serial_conn.write(command_bytes)
            logger.debug(f"Sent command: {command}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {str(e)}")
            return False
    
    def read_response(self, timeout: float = 5.0) -> Optional[str]:
        """
        Read response from STM32
        
        Args:
            timeout: Read timeout in seconds
            
        Returns:
            Response string or None if timeout/error
        """
        if not self.is_connected or not self.serial_conn:
            return None
        
        try:
            start_time = time.time()
            response_lines = []
            
            while (time.time() - start_time) < timeout:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:
                        response_lines.append(line)
                        logger.debug(f"Received: {line}")
                        
                        # Check for command completion
                        if any(keyword in line for keyword in ["OK:", "ERROR:", "STATUS:"]):
                            return '\n'.join(response_lines)
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
            
            logger.warning("Response timeout")
            return None
            
        except Exception as e:
            logger.error(f"Failed to read response: {str(e)}")
            return None
    
    def collect_sample(self, fault_type: MotorFaultType, 
                      sample_description: str = "") -> Optional[MotorSample]:
        """
        Collect a single vibration sample from the motor
        
        Args:
            fault_type: Type of motor fault to collect
            sample_description: Optional description for the sample
            
        Returns:
            MotorSample object or None if collection failed
        """
        if not self.is_connected:
            logger.error("Not connected to STM32")
            return None
        
        # Map fault type to command
        command_map = {
            MotorFaultType.NORMAL: "START_NORMAL",
            MotorFaultType.IMBALANCE: "START_IMBALANCE",
            MotorFaultType.BEARING_FAULT: "START_BEARING",
            MotorFaultType.MISALIGNMENT: "START_MISALIGN"
        }
        
        command = command_map.get(fault_type)
        if not command:
            logger.error(f"Unknown fault type: {fault_type}")
            return None
        
        logger.info(f"Starting data collection for {fault_type.name}")
        
        # Start collection
        if not self.send_command(command):
            return None
        
        response = self.read_response()
        if not response or "OK:" not in response:
            logger.error(f"Failed to start collection: {response}")
            return None
        
        # Wait for collection to complete and data to be ready
        logger.info("Collection started, waiting for completion...")
        sample_data = self.wait_for_sample_data()
        
        if sample_data:
            sample_data.metadata.update({
                'description': sample_description,
                'collection_time': datetime.now().isoformat()
            })
            self.collected_samples.append(sample_data)
            logger.info(f"Successfully collected sample {sample_data.sample_id}")
            return sample_data
        else:
            logger.error("Failed to collect sample data")
            return None
    
    def wait_for_sample_data(self, max_wait_time: float = 30.0) -> Optional[MotorSample]:
        """
        Wait for sample data transmission from STM32
        
        Args:
            max_wait_time: Maximum time to wait for data
            
        Returns:
            MotorSample object or None if timeout/error
        """
        start_time = time.time()
        buffer = ""
        in_data_section = False
        sample_info = {}
        data_lines = []
        
        while (time.time() - start_time) < max_wait_time:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8').strip()
                
                if line == "AI_SAMPLE_START":
                    logger.info("Sample data transmission started")
                    continue
                elif line == "DATA_START":
                    in_data_section = True
                    continue
                elif line == "DATA_END":
                    in_data_section = False
                    continue
                elif line == "AI_SAMPLE_END":
                    logger.info("Sample data transmission completed")
                    break
                elif line.startswith("ID:"):
                    sample_info['sample_id'] = int(line.split(':')[1])
                elif line.startswith("TIMESTAMP:"):
                    sample_info['timestamp'] = int(line.split(':')[1])
                elif line.startswith("FAULT_TYPE:"):
                    sample_info['fault_type'] = MotorFaultType(int(line.split(':')[1]))
                elif line.startswith("SAMPLE_RATE:"):
                    sample_info['sample_rate'] = int(line.split(':')[1])
                elif line.startswith("DURATION:"):
                    sample_info['duration_ms'] = int(line.split(':')[1])
                elif line.startswith("NUM_SAMPLES:"):
                    sample_info['num_samples'] = int(line.split(':')[1])
                elif in_data_section and ',' in line:
                    data_lines.append(line)
            
            time.sleep(0.01)
        
        # Parse collected data
        if data_lines and sample_info:
            return self.parse_sample_data(sample_info, data_lines)
        else:
            logger.error("Incomplete sample data received")
            return None
    
    def parse_sample_data(self, sample_info: Dict, data_lines: List[str]) -> MotorSample:
        """
        Parse raw sample data into MotorSample object
        
        Args:
            sample_info: Sample metadata
            data_lines: Raw data lines in format "x,y,z"
            
        Returns:
            MotorSample object
        """
        x_data, y_data, z_data = [], [], []
        
        for line in data_lines:
            try:
                x, y, z = map(int, line.split(','))
                x_data.append(x)
                y_data.append(y)
                z_data.append(z)
            except ValueError:
                logger.warning(f"Skipping invalid data line: {line}")
        
        return MotorSample(
            sample_id=sample_info['sample_id'],
            timestamp=sample_info['timestamp'],
            fault_type=sample_info['fault_type'],
            sample_rate=sample_info['sample_rate'],
            duration_ms=sample_info['duration_ms'],
            num_samples=len(x_data),
            x_data=x_data,
            y_data=y_data,
            z_data=z_data,
            metadata={}
        )
    
    def save_samples_to_file(self, filename: str, file_format: str = 'hdf5'):
        """
        Save collected samples to file
        
        Args:
            filename: Output filename
            file_format: File format ('hdf5', 'csv', 'json')
        """
        if not self.collected_samples:
            logger.warning("No samples to save")
            return
        
        if file_format == 'hdf5':
            self.save_to_hdf5(filename)
        elif file_format == 'csv':
            self.save_to_csv(filename)
        elif file_format == 'json':
            self.save_to_json(filename)
        else:
            logger.error(f"Unsupported file format: {file_format}")
    
    def save_to_hdf5(self, filename: str):
        """Save samples to HDF5 format (recommended for large datasets)"""
        import h5py
        
        with h5py.File(filename, 'w') as f:
            for i, sample in enumerate(self.collected_samples):
                group = f.create_group(f'sample_{sample.sample_id}')
                group.attrs['fault_type'] = sample.fault_type.value
                group.attrs['sample_rate'] = sample.sample_rate
                group.attrs['duration_ms'] = sample.duration_ms
                group.attrs['timestamp'] = sample.timestamp
                
                group.create_dataset('x_data', data=np.array(sample.x_data))
                group.create_dataset('y_data', data=np.array(sample.y_data))
                group.create_dataset('z_data', data=np.array(sample.z_data))
        
        logger.info(f"Saved {len(self.collected_samples)} samples to {filename}")
    
    def save_to_csv(self, filename: str):
        """Save samples to CSV format"""
        all_data = []
        
        for sample in self.collected_samples:
            for i in range(len(sample.x_data)):
                all_data.append({
                    'sample_id': sample.sample_id,
                    'fault_type': sample.fault_type.value,
                    'fault_name': sample.fault_type.name,
                    'sample_index': i,
                    'x': sample.x_data[i],
                    'y': sample.y_data[i],
                    'z': sample.z_data[i],
                    'timestamp': sample.timestamp,
                    'sample_rate': sample.sample_rate
                })
        
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(self.collected_samples)} samples to {filename}")


def main():
    """Example usage of the data collector"""
    
    # Configuration
    SERIAL_PORT = "COM3"  # Change this to your STM32 port
    OUTPUT_DIR = "collected_data"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize collector
    collector = STM32DataCollector(SERIAL_PORT)
    
    try:
        # Connect to STM32
        if not collector.connect():
            logger.error("Failed to connect to STM32")
            return
        
        # Collection plan
        collection_plan = [
            (MotorFaultType.NORMAL, "Healthy motor baseline"),
            (MotorFaultType.IMBALANCE, "Motor with added weight imbalance"),
            (MotorFaultType.BEARING_FAULT, "Motor with bearing wear"),
            (MotorFaultType.MISALIGNMENT, "Motor shaft misalignment")
        ]
        
        # Collect samples
        for fault_type, description in collection_plan:
            print(f"\n=== Collecting {fault_type.name} samples ===")
            print(f"Description: {description}")
            input("Press Enter when motor is ready...")
            
            # Collect multiple samples for this fault type
            for i in range(3):  # 3 samples per fault type
                sample = collector.collect_sample(fault_type, f"{description} - Sample {i+1}")
                if sample:
                    print(f"✓ Collected sample {sample.sample_id}")
                else:
                    print("✗ Failed to collect sample")
                
                time.sleep(2)  # Wait between samples
        
        # Save collected data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"motor_data_{timestamp}.h5")
        collector.save_samples_to_file(output_file, 'hdf5')
        
        print(f"\nData collection complete! Saved to {output_file}")
        print(f"Total samples collected: {len(collector.collected_samples)}")
        
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        collector.disconnect()


if __name__ == "__main__":
    main()
