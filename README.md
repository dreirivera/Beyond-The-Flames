# Beyond the Flames: Fire Behavior Simulation with Machine Learning

This repository contains the thesis project **"Beyond the Flames: Exploring the Role of Oxygen Concentration, Wind Speed, Humidity, and Temperature in Fire Behavior"**, developed at Cavite State University.  

The project integrates **machine learning (Random Forest regression)** with **Fire Dynamics Simulator (FDS)** and **Smokeview (SMV)** by NIST to predict **fire intensity** and **heat release rate (HRR)**, while identifying the most influential environmental variable.

---

## 📖 Abstract
Fire remains one of the most complex and dynamic physical phenomena due to the interaction of numerous environmental variables. This study presents a **machine learning–based fire simulation model** that predicts fire intensity and heat release rate (HRR) using Random Forest regression. Among the four variables tested—oxygen concentration, wind speed, humidity, and temperature—**temperature was found to be the most impactful**.  

Predicted HRR values are integrated with **FDS/SMV** for real-time fire visualization, improving simulation efficiency and fire risk analysis.

---

## ⚙️ System Requirements
**Hardware**
- Processor: Intel i5 / Ryzen 5 or better
- RAM: 8 GB
- Storage: 1 GB free space
- Display: 1366x768 or higher

**Software**
- Windows 10+, macOS 10.15+, or Ubuntu 20.04+
- Python 3.9+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tkinter`
- [FDS](https://pages.nist.gov/fds-smv/) and [Smokeview](https://pages.nist.gov/fds-smv/) installed and added to PATH

---

## 🚀 Installation
1. Install **Python 3.9+** and required libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Install **FDS** and **Smokeview**, then add them to your system PATH.
   - Example (Windows default path):  
     `C:\Program Files\firemodels\FDS6\bin`
3. Restart your terminal/IDE to apply PATH changes.
4. Test installation:
   ```bash
   fds
   smokeview -help
   ```

---

## ▶️ Usage
1. Launch the program:
   ```bash
   gui.py
   ```
2. Input values for:
   - Oxygen Concentration (%)
   - Temperature (°C)
   - Wind Speed (km/h)
   - Humidity (%)
3. Click **Simulate** to generate predictions:
   - Fire Intensity (kW/m²)  
   - Heat Release Rate (HRR) (kW)  
   - Most Influential Variable  
   - Visualization in **Smokeview (SMV)**

---

## 📊 Outputs
- Predicted Fire Intensity  
- Predicted HRR  
- Feature importance ranking of input variables  
- Line graphs of MSE and  R² Metric
- 3D Smokeview visualization of fire behavior  

---

## 🛠 Troubleshooting
- **GUI doesn’t reopen after closing:**  
  The Smokeview process may still be running. Stop it manually from your IDE/terminal before relaunching.
- Always use **"Simulate Another Scenario"** instead of closing the app with the (X) button.

---

## 👩‍💻 Author
- **Edrei Reigne I. Rivera** – edreireigne.rivera@cvsu.edu.ph  

Institution: **Cavite State University**  
College of Engineering and Information Technology  
Department of Information Technology

---

## 📜 License
This repository uses a **dual license**:
- **Thesis (PDF, documentation, datasets)** → [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)  
- **Code/scripts** → [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

See the [LICENSE](./LICENSE) file for full details.
