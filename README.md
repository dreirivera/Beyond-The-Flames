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

## ⚙️ Step-by-Step Setup

### Step 1: Locate the FDS Installation Folder
1. Open **File Explorer**.  
2. Navigate to the folder where FDS was installed.  
   - Default location:  
     ```
     C:\Program Files\firemodels\FDS6\bin
     ```  
   *(Path may vary depending on your installation.)*  
3. Inside this folder, confirm that you see:
   - `fds.exe`  
   - `smokeview.exe`  

### Step 2: Copy the Folder Path
- In File Explorer, click the address bar at the top and copy the path.

### Step 3: Add FDS to the System PATH
1. Press **Windows + S**, search for *Environment Variables*, and select:  
   **Edit the system environment variables**  
2. In the *System Properties* window, click **Environment Variables…**.  
3. Under *System variables*, find and select the variable named **Path**.  
4. Click **Edit**, then **New**.  
5. Paste the path you copied earlier.  
6. Click **OK** on all windows to save changes.

### Step 4: Restart CMD or Your Python IDE
- Any open terminal or IDE must be restarted so the new PATH takes effect.

### Step 5: Test from Command Prompt
1. Open a regular **Command Prompt** (not the FDS version).  
2. Type:
   ```bash
   fds
   ```
   You should see the FDS information.  
3. Type:
   ```bash
   smokeview -help
   ```
   You should see Smokeview usage info.  

✅ If both commands return output, you’re all set!  

*(Optional)*: Run the included `test_script.py` to check if both FDS and SMV are working.

---

## ▶️ Running a Simulation

1. Slide the meters to set values for all four variables:
   - **Oxygen Concentration** (%)
   - **Temperature** (°C)
   - **Wind Speed** (km/h)
   - **Humidity** (%)
2. Click **Run Simulation**.  
3. The model will process your input using a trained Random Forest regression model.  
4. The system will output:
   - Fire Intensity (kW/m²)  
   - Heat Release Rate (HRR) (kW)  
   - Most Influential Variable  

⚠️ Note: You must run at least **5 simulations** before the model calculates the most influential variable.

5. The **SMV window** will appear with a 3D simulation box.

---

## 🔥 Viewing Simulation Results in Smokeview (SMV)

1. Click the **SMV window**.  
2. Right-click, hover **Load/Unload**.  
3. Hover **3D Smoke**.  
4. Select one or more options:
   - **Soot Density** → shows smoke/opacity  
   - **HRRPUV** → shows combustion intensity  
   - **Temperature** → shows heat distribution  
5. To display HRR values:
   - Right-click → hover **Show/Hide → Labels → HRR**  
   - Make sure one of *Soot Density*, *HRRPUV*, or *Temperature* is active.

---

## 🛠 Troubleshooting

### Problem: GUI won’t launch after closing the app
- After closing the Fire Simulation App, re-running the code does not reopen the GUI.  

**Cause:**  
The Smokeview (SMV) process may not have terminated properly, so the OS still thinks it’s running.

**Fix:**  
1. Check the terminal/console in your IDE (e.g., VS Code, PyCharm).  
   - You’ll see the program is still running even if the GUI is closed.  
2. Manually stop the process:  
   - In VS Code → click the red ❌ **Terminate** button  
   - Or press **Ctrl + C** in the terminal  

✅ Once stopped, re-run the code. The GUI should now open normally.

**Prevention Tips:**  
- Always wait for the simulation to finish before closing the app.  
- Use the **"Simulate Another Scenario"** button instead of the close (X) button during simulation.  

---

## 👩‍💻 Author
- **Edrei Reigne I. Rivera** – edreireigne.rivera@cvsu.edu.ph  

Institution: **Cavite State University**  
College of Engineering and Information Technology  
Department of Information Technology

---

## 📜 License
This repository uses a **dual license**:
- **Thesis (datasets)** → [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)  
- **Code/scripts** → [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

See the [LICENSE](./LICENSE) file for full details.
