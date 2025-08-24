# Edge AI Motor Anomaly Detection on STM32H745ZI

A dual-core (CM7/CM4) embedded system that detects electric motor anomalies on-device using a 3-axis accelerometer and an int8 neural network. CM4 acquires data and shares it via a ring buffer in D2 SRAM; CM7 runs inference using X-CUBE-AI and drives LED/buzzer feedback.

## Features
- STM32H745ZI dual-core architecture (CM7 inference, CM4 acquisition)
- MSA301 3-axis accelerometer over I²C
- QSPI NOR flash (W25Q256JV) for logs/models
- TIM6-based 1 kHz data collection for AI dataset capture
- Quantized (int8) CNN deployed via STM32Cube.AI (X-CUBE-AI)
- Shared D2 SRAM ring buffer (.shared_ram) for inter-core exchange
- LED/buzzer feedback for anomaly indication
- Python pipeline for data collection, preprocessing, training, quantization

## Repository Structure
