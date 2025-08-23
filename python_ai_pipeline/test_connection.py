#!/usr/bin/env python3
"""
Simple test script to verify STM32 connection and basic functionality
"""

import serial
import time
import sys

def test_stm32_connection(port, baudrate=115200):
    """Test basic connection to STM32 board"""
    
    print(f"Testing connection to STM32 on port {port}...")
    
    try:
        # Open serial connection
        ser = serial.Serial(port, baudrate, timeout=5)
        time.sleep(2)  # Wait for connection to stabilize
        
        # Clear any existing data
        ser.flushInput()
        ser.flushOutput()
        
        print("✓ Serial connection established")
        
        # Test basic communication
        print("Testing basic communication...")
        ser.write(b"STATUS\r\n")
        
        # Read response
        response_lines = []
        start_time = time.time()
        
        while (time.time() - start_time) < 5:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    print(f"Received: {line}")
                    response_lines.append(line)
                    
                    if "STATUS" in line or "OK:" in line or "ERROR:" in line:
                        break
            time.sleep(0.1)
        
        if response_lines:
            print("✓ Communication successful")
        else:
            print("✗ No response received")
            return False
        
        # Test data collection command
        print("\nTesting data collection command...")
        ser.write(b"START_NORMAL\r\n")
        
        response_lines = []
        start_time = time.time()
        
        while (time.time() - start_time) < 5:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    print(f"Received: {line}")
                    response_lines.append(line)
                    
                    if "OK:" in line or "ERROR:" in line:
                        break
            time.sleep(0.1)
        
        # Stop collection
        time.sleep(1)
        ser.write(b"STOP\r\n")
        
        # Read stop response
        start_time = time.time()
        while (time.time() - start_time) < 3:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    print(f"Received: {line}")
                    if "OK:" in line or "ERROR:" in line:
                        break
            time.sleep(0.1)
        
        ser.close()
        print("✓ Connection test completed successfully")
        return True
        
    except serial.SerialException as e:
        print(f"✗ Serial connection error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    
    # Common serial ports to try
    if len(sys.argv) > 1:
        ports_to_test = [sys.argv[1]]
    else:
        # Default ports for different systems
        ports_to_test = [
            "COM3", "COM4", "COM5",  # Windows
            "/dev/ttyUSB0", "/dev/ttyUSB1",  # Linux
            "/dev/ttyACM0", "/dev/ttyACM1",  # Linux (CDC)
        ]
    
    print("STM32H745 Connection Test")
    print("=" * 30)
    
    success = False
    for port in ports_to_test:
        print(f"\nTrying port: {port}")
        try:
            if test_stm32_connection(port):
                success = True
                print(f"\n✓ Successfully connected to STM32 on port {port}")
                break
        except Exception as e:
            print(f"✗ Failed to connect to {port}: {e}")
            continue
    
    if not success:
        print("\n✗ Could not establish connection to STM32")
        print("\nTroubleshooting tips:")
        print("1. Check that STM32 board is connected via USB")
        print("2. Verify the correct COM port in Device Manager (Windows)")
        print("3. Ensure STM32 firmware is flashed and running")
        print("4. Try a different USB cable")
        print("5. Check power supply to the board")
        return 1
    
    print("\n" + "=" * 50)
    print("Connection test completed successfully!")
    print("You can now use data_collector.py to collect training data.")
    return 0

if __name__ == "__main__":
    exit(main())
