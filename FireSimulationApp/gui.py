from ml_model import predict_fire_behavior, update_user_simulations, get_model_metrics
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import csv
import time
import threading
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class FireSimulationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fire Simulation Model")
        self.geometry("1000x700")
        self.fds_process = None
        self.smv_process = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.config(bg="#fae78d")

        self.frames = {}
        for F in (HomePage, ProgressPage, ResultsPage):
            frame = F(parent=self, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            frame.config(bg="#fae78d")

        self.show_frame("HomePage")

    def show_frame(self, page_name):
        if page_name == "HomePage":
            if self.fds_process and self.fds_process.poll() is None:
                self.fds_process.terminate()
            if self.smv_process and self.smv_process.poll() is None:
                self.smv_process.terminate()
            self.frames["HomePage"].reset_inputs()
        self.frames[page_name].tkraise()

    def on_close(self):
        if self.fds_process and self.fds_process.poll() is None:
            self.fds_process.terminate()
        if self.smv_process and self.smv_process.poll() is None:
            self.smv_process.terminate()
        if os.path.exists("user_simulations.csv"):
            os.remove("user_simulations.csv")
        self.destroy()


class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        title_font = ("Helvetica", 18, "bold")
        label_font = ("Helvetica", 12)
        button_font = ("Helvetica", 12, "bold")

        tk.Label(self, text="Enter Fire Simulation Variables", font=title_font, bg="#fae78d", fg="#222").pack(pady=20)

        self.entries = {}
        self.ranges = {
            "Temperature (20-40°C)": (20, 40),
            "Wind Speed (5-20 km/h)": (5, 20),
            "Humidity (20-80%)": (20, 80),
            "Oxygen Concentration (15-30%)": (15, 30)
        }

        for var, (min_val, max_val) in self.ranges.items():
            frame = tk.Frame(self, bg="#fae78d")
            frame.pack(pady=5)
            tk.Label(frame, text=var, width=35, anchor='w', font=label_font, bg="#fae78d", fg="#222").pack(side="left")
            scale = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                             resolution=1, length=300, font=label_font, bg="#fae78d", fg="#222")
            scale.set((min_val + max_val) // 2)
            scale.pack(side="left")
            self.entries[var] = scale

        tk.Button(self, text="Run Simulation", command=self.run_simulation, bg="#4CAF50", fg="white",
                  width=20, font=button_font, relief="raised").pack(pady=20)

    def run_simulation(self):
        user_inputs = {}
        for var, scale in self.entries.items():
            value = scale.get()
            min_val, max_val = self.ranges[var]
            if not (min_val <= value <= max_val):
                messagebox.showerror("Invalid Input", f"{var} must be between {min_val} and {max_val}.")
                return
            user_inputs[var] = value

        user_inputs_for_ml = {
            "temperature": user_inputs["Temperature (20-40°C)"],
            "wind_speed": user_inputs["Wind Speed (5-20 km/h)"],
            "humidity": user_inputs["Humidity (20-80%)"],
            "oxygen": user_inputs["Oxygen Concentration (15-30%)"],
        }

        self.controller.user_inputs = user_inputs_for_ml
        self.controller.user_inputs_full = user_inputs

        predicted_intensity, predicted_hrr, influential_variable = predict_fire_behavior(user_inputs_for_ml)

        self.generate_fds_file(user_inputs, predicted_hrr)

        self.controller.show_frame("ProgressPage")
        self.controller.frames["ProgressPage"].start_simulation()

    def generate_fds_file(self, inputs, predicted_hrr):
        hrr = max(predicted_hrr, 1)

        fds_template = f"""
    &HEAD CHID='fire_sim', TITLE='Dynamic Fire Simulation' /
    &MESH IJK=20,21,15, XB=0.0,20.0, 0.0,21.0, 0.0,15.0 /
    &TIME T_END=60.0 /
    &MATL ID='WALL_MATERIAL', CONDUCTIVITY=0.5, DENSITY=800.0, SPECIFIC_HEAT=1.5 /
    &SURF ID='WALL', COLOR='BLACK', MATL_ID='WALL_MATERIAL', THICKNESS=0.1 /
    &SURF ID='FIRE', HRRPUA={hrr:.2f}, COLOR='RED' /

    &VENT XB=0.0,20.0,0.0,0.0,0.0,15.0, SURF_ID='OPEN' /
    &VENT XB=0.0,20.0,21.0,21.0,0.0,15.0, SURF_ID='OPEN' /

    &REAC ID='DIESEL_REACTION', FUEL='PROPANE', CO_YIELD=0.05, SOOT_YIELD=0.015 /

    &OBST XB=7.0,13.0,8.0,14.0,0.0,2.5, SURF_ID='FIRE' /
    &OBST XB=9.5,10.5,10.5,11.5,0.0,0.75, SURF_ID='WALL' /

    &TAIL /
    """

        with open("fire_sim.fds", "w", encoding="utf-8") as f:
            f.write(fds_template)

        with open("fire_sim.ini", "w", encoding="utf-8") as f:
            f.write("HRRPUV_min=0.5\n")

    def reset_inputs(self):
        for scale in self.entries.values():
            min_val = scale.cget("from")
            max_val = scale.cget("to")
            scale.set((min_val + max_val) // 2)



class ProgressPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#fae78d")
        # For showing simulation progress and status
        self.controller = controller
        
        title_font = ("Helvetica", 18, "bold")
        label_font = ("Helvetica", 14)

        tk.Label(self, text="Simulation Progress", font=title_font, bg="#fae78d", fg="#222").pack(pady=20)
        self.progress_label = tk.Label(self, text="Simulation being generated 0/100%", font=label_font, bg="#fae78d", fg="#222")
        self.progress_label.pack(pady=10)
        self.progress = ttk.Progressbar(self, length=400, mode='determinate')
        self.progress.pack(pady=20)
        self.loading = True
        
        
    def start_simulation(self):
        # For starting the FDS simulation in a background thread
        self.progress["value"] = 0
        self.loading = True
        self.update_progress(0)
        threading.Thread(target=self.run_fds_simulation, daemon=True).start()

    def update_progress(self, value):
        # For updating the progress bar animation
        if value < 98 and self.loading:
            self.progress["value"] = value
            self.progress_label.config(text=f"Simulation being generated {value}/100%")
            self.after(1200, self.update_progress, value + 2)

    def run_fds_simulation(self):
        # For executing FDS simulation and waiting for SMV file
        env = os.environ.copy()
        env["I_MPI_ROOT"] = r"C:\\Program Files\\firemodels\\FDS6\\bin\\mpi"
        env["PATH"] = r"C:\\Program Files\\firemodels\\FDS6\\bin\\mpi;" + env["PATH"]
        self.controller.fds_process = subprocess.Popen(["fds", "fire_sim.fds"], env=env)
        self.controller.fds_process.wait()

        while not os.path.exists("fire_sim.smv"):
            time.sleep(1)

        self.loading = False
        self.after(100, self.finalize_progress)

    def finalize_progress(self):
        # For finishing progress and moving to the results page
        self.progress["value"] = 100
        self.progress_label.config(text="Simulation being generated 100/100%")
        self.after(500, lambda: self.controller.show_frame("ResultsPage"))
        self.after(500, lambda: self.controller.frames["ResultsPage"].display_results())

class ResultsPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#fae78d")
        # For showing predicted results and accuracy charts
        self.controller = controller

        title_font = ("Helvetica", 18, "bold")
        label_font = ("Helvetica", 14)
        button_font = ("Helvetica", 12, "bold")

        tk.Label(self, text="Simulation Results", font=title_font, bg="#fae78d", fg="#222").pack(pady=10)

        results_frame = tk.Frame(self, bg="#fae78d")
        results_frame.pack(fill="both", expand=True)

        self.prediction_frame = tk.Frame(results_frame, bg="#fae78d", width=450, height=400)
        self.prediction_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.predicted_intensity_label = tk.Label(self.prediction_frame, text="Fire Intensity: N/A", font=label_font, bg="#fae78d", fg="#222")
        self.predicted_intensity_label.pack(pady=10)
        self.predicted_hrr_label = tk.Label(self.prediction_frame, text="Heat Release Rate: N/A", font=label_font, bg="#fae78d", fg="#222")
        self.predicted_hrr_label.pack(pady=10)
        self.simulation_number_label = tk.Label(self.prediction_frame, text="Number of Simulation: N/A", font=label_font, bg="#fae78d", fg="#222")
        self.simulation_number_label.pack(pady=5)
        self.predicted_influencial_label = tk.Label(self.prediction_frame, text="Most Influential Variable: N/A", font=label_font, bg="#fae78d", fg="#222")
        self.predicted_influencial_label.pack(pady=5)

        self.graph_frame = tk.Frame(results_frame, bg="#fae78d", width=450)
        self.graph_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        tk.Button(self, text="Simulate Another Scenario", command=lambda: controller.show_frame("HomePage"),
                  bg="#4CAF50", fg="white", width=25, font=button_font, relief="raised").pack(pady=20)
        
        
    def display_results(self):
        # For launching Smokeview and updating the results display
        self.controller.smv_process = subprocess.Popen(["smokeview", "fire_sim"])
        inputs = self.controller.user_inputs
        predicted_intensity, predicted_hrr, influential_variable = predict_fire_behavior(inputs)
        simulation_number = update_user_simulations(inputs, predicted_intensity, predicted_hrr, influential_variable)

        self.predicted_intensity_label.config(text=f"Fire Intensity: {round(predicted_intensity, 2)}")
        self.predicted_hrr_label.config(text=f"Heat Release Rate: {round(predicted_hrr, 2)} kW")
        self.simulation_number_label.config(text=f"Number of Simulation: {simulation_number}")
        self.predicted_influencial_label.config(text=f"Most Influential Variable: {influential_variable}")

        self.plot_model_metrics()
        def on_closing(self):
            if self.smv_process and self.smv_process.poll() is None:
                self.smv_process.terminate()
                self.smv_process.wait()
            self.destroy()
            sys.exit()

    def plot_model_metrics(self):
        # For displaying MSE and R² line graphs of model accuracy

        def update_graph(selection):
            # For clearing any existing graph before plotting a new one
            for widget in self.graph_canvas_frame.winfo_children():
                widget.destroy() 

            # For creating a new figure and axis for the plot
            fig, ax = plt.subplots(figsize=(6, 4))

            # For selecting the correct data and labels based on user dropdown choice
            if selection == "Intensity (R²)":
                train_data = r2_histories["Intensity Train"]
                validate_data = r2_histories["Intensity Validation"]
                ylabel = "R² Score"
                ylim = (0, 1)
            elif selection == "Intensity (MSE)":
                train_data = mse_histories["Intensity Train"]
                validate_data = mse_histories["Intensity Validation"]
                ylabel = "MSE Score"
                ylim = None
            elif selection == "HRR (R²)":
                train_data = r2_histories["HRR Train"]
                validate_data = r2_histories["HRR Validation"]
                ylabel = "R² Score"
                ylim = (0, 1)
            elif selection == "HRR (MSE)":
                train_data = mse_histories["HRR Train"]
                validate_data = mse_histories["HRR Validation"]
                ylabel = "MSE Score" 
                ylim = None
            else:
                return

            # For plotting the train and validation data as line graphs
            ax.plot(run_ids, train_data, marker='o', linestyle='-', label="Train")
            ax.plot(run_ids, validate_data, marker='o', linestyle='-', label="Validation")

            # For calculating vertical offsets for the value labels
            offset_train = (max(train_data) * 0.02 if max(train_data) > 0 else 0.01)
            offset_test = (max(validate_data) * 0.02 if max(validate_data) > 0 else 0.01)

            # For setting axis labels, title, limits, and styles
            ax.set_title(f"{selection} Over Time")
            ax.set_xlabel("Increment")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=8)
            
            # For improving layout spacing
            fig.tight_layout(pad=2.0)

            # For embedding the Matplotlib plot into the Tkinter GUI
            canvas = FigureCanvasTkAgg(fig, master=self.graph_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        # For clearing previous dropdowns or widgets
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        try:
            # For loading all past model metrics from CSV
            metrics = get_model_metrics()

            r2_keys = ["r2_intensity_train", "r2_intensity_validation", "r2_hrr_train", "r2_hrr_validation"]
            mse_keys = ["mse_intensity_train", "mse_intensity_validation", "mse_hrr_train", "mse_hrr_validation"]
            labels = ["Intensity Train", "Intensity Validation", "HRR Train", "HRR Validation"]

            history_file = "accuracy_history.csv"
            header = ["Increment"] + r2_keys + mse_keys

            history = []
            if os.path.exists(history_file):
                with open(history_file, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        history.append(row)

            # For extracting run IDs and metric histories into lists
            run_ids = [int(float(row["run_id"])) for row in history]
            r2_histories = {label: [] for label in labels}
            mse_histories = {label: [] for label in labels}

            for row in history:
                for i, label in enumerate(labels):
                    r2_val = float(row[r2_keys[i]])
                    mse_val = float(row[mse_keys[i]])
                    r2_histories[label].append(r2_val)
                    mse_histories[label].append(mse_val)

            # For creating the metric selection dropdown
            dropdown_label = tk.Label(self.graph_frame, text="Select Metric & Variable:")
            dropdown_label.pack(pady=5)

            selected_option = tk.StringVar()
            dropdown = ttk.Combobox(self.graph_frame, textvariable=selected_option, state="readonly")
            dropdown['values'] = ["Intensity (R²)", "Intensity (MSE)", "HRR (R²)", "HRR (MSE)"]
            dropdown.current(0)
            dropdown.pack(pady=5)

            # For preparing the frame to hold the chart canvas
            self.graph_canvas_frame = tk.Frame(self.graph_frame)
            self.graph_canvas_frame.pack(fill="both", expand=True)

            # For drawing the initial plot and handling dropdown changes
            update_graph(dropdown.get())
            dropdown.bind("<<ComboboxSelected>>", lambda e: update_graph(dropdown.get()))

        except Exception as e:
            print("Error plotting model metrics:", e)



if __name__ == "__main__":
    app = FireSimulationApp()
    app.mainloop()
    sys.exit()