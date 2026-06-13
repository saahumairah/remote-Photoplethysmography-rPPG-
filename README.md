# Remote Photoplethysmography (rPPG) - Contactless Heart Rate Monitor

This repository contains the software implementation for a non-invasive heart rate monitoring system based on medical image processing. This system utilizes a standard RGB camera to record and analyze microscopic fluctuations in facial skin color caused by variations in blood volume within the microvascular tissue during the cardiovascular cycle.

## Background and Technological Solution
Conventional vital sign monitoring typically relies on contact-based methods, such as electrocardiograms or pulse oximeters, which can pose risks of irritation or discomfort to patients during long-term monitoring scenarios. This project offers an alternative solution by developing a remote Photoplethysmography (rPPG) method. To overcome the primary challenges of subject movement artifacts and environmental illumination variations, this system specifically implements the Plane-Orthogonal-to-Skin (POS) algorithm. This mathematical approach works by projecting the RGB signals onto an orthogonal plane to isolate pure pulse fluctuations from specular reflection noise.

## Digital Signal Processing Architecture
The computational process within this program runs in real-time through a rigorous series of signal processing stages. Initially, the face detection system is stabilized using an Exponential Moving Average (EMA) filter to prevent spatial jitter on the Region of Interest (ROI) in the forehead area. The captured color signals are then filtered in the time domain using a 5th-Order Butterworth Bandpass Filter to block low-frequency respiratory signals and high-frequency camera noise. Subsequently, the signals are transformed into the frequency domain using the Fast Fourier Transform (FFT). The resulting spectrum is then multiplied by a Gaussian probability distribution weighting (A-Weighting) to dampen low-frequency dominance and estimate the heart rate value with high precision.

## System Performance Evaluation
This rPPG system has been directly validated by comparing its heart rate estimation output against a digital pulse oximeter, which serves as the medical standard. Based on a series of tests, this software demonstrates a highly satisfactory level of accuracy, recording a Mean Absolute Error (MAE) of 3.3 BPM and a Mean Absolute Percentage Error (MAPE) of 3.65%. This deviation rate, which falls well below the 5% tolerance threshold, proves the viability of this algorithm as a reliable contactless vital sign monitoring modality.

## Installation and Usage Guide
This project is developed entirely using the Python programming language. To replicate and run this image processing script on a local computer, you are required to install several scientific computing libraries. Please open your terminal or command prompt application, then execute the following installation command:

`pip install opencv-python numpy scipy`

Once all supporting libraries are successfully installed, you can directly launch the system interface by executing the main file. Ensure you are in an environment with stable room lighting and face the device's camera directly. Execute the program using the following command:

`python rPPG_by_sahum.py`

## Project Developer
This system was designed and developed by Khalisa Humairah as part of the final project for the Medical Image Processing course in the Department of Biomedical Engineering, Faculty of Intelligent Electrical and Informatics Technology, Institut Teknologi Sepuluh Nopember.
