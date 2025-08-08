# Detecting Body-Focused Repetitive Behaviors (BFRBs) with Sensor Data

## Overview

This repository documents my journey through a **machine learning competition** focused on detecting **Body-Focused Repetitive Behaviors (BFRBs)** — self-directed habits like:

- Hair pulling
- Skin picking
- Nail biting

While these behaviors are often harmless, when frequent or intense, they can cause **physical harm** and **psychosocial challenges**. They’re common in anxiety disorders and OCD, making them important indicators for mental health.

The **goal** of this project is to detect BFRB-like gestures from **wrist-worn sensor data**, captured by the [Helios device](https://childmind.org) developed by the Child Mind Institute.

---

## The Helios Device

The **Helios** device is not your average smartwatch — it combines **three types of sensors**:

| Sensor                        | Description                                                 | Why it matters                               |
| ----------------------------- | ----------------------------------------------------------- | -------------------------------------------- |
| **IMU (BNO080/BNO085)**       | Measures acceleration, rotation, and orientation in 3D.     | Detects movement patterns.                   |
| **Thermopile (MLX90632)**     | 5 temperature sensors measuring infrared radiation.         | Detects proximity to warm skin areas.        |
| **Time-of-Flight (VL53L7CX)** | 5 distance sensors with 8×8 pixel grids (64 readings each). | Maps proximity and object location in space. |

This gives **a lot** of data per gesture.  
For example, **Time-of-Flight alone** produces `5 × 64 = 320` features **per time step**!

---

## The Dataset

The dataset contains **sensor recordings** of participants performing:

- **8 BFRB-like gestures** (target class)
- **10 non-BFRB gestures** (non-target class)

Each recording follows this structure:

1. **Transition** → moving from rest to gesture position.
2. **Pause** → short still moment.
3. **Gesture** → perform the movement.

**Files included**:

- `train.csv` / `test.csv` → sensor readings per time step.
- `*_demographics.csv` → participant info (age, sex, handedness, height, etc.).

  **Note:** About half of the test sequences contain only IMU data — thermopile and ToF are missing due to hardware failure or whatever.

---

## Challenge

1. **High dimensionality** — The ToF data alone can produce **64,000+ values** for a single sequence.
2. **Missing data** — ToF readings often contain `-1` (no object detected).
3. **Variable sequence length** — Gestures vary in duration, so each `sequence_id` contains a different number of time steps.

---
