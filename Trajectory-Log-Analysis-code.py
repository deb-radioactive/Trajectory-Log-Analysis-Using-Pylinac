from pylinac import TrajectoryLog
import matplotlib.pyplot as plt
import numpy as np

log_path = r"C:\Users\debri\Desktop\SRS_SRT\S-0462-25_FSRS30Gy5Fr_Arc1-CW-Brain_20250526115007.bin"
tlog = TrajectoryLog(log_path)

# Extract header information
header = tlog.header
print("\n=== Header Information ===")
print(f"Log Version: {header.version}")
print(f"Header Size: {header.header_size}")
print(f"Sampling Interval(msec): {header.sampling_interval}")
print(f"Number of axes: {header.num_axes}")
print(f"Number of MLC Leaves: {header.num_mlc_leaves}")
print(f"Number of Subbeams: {header.num_subbeams}")
print(f"Number of Snapshots: {header.num_snapshots}")

# Beam Holdoff and Beam On/Off Analysis
print("\n=== Beam Holdoff Analysis ===")
beam_holds = tlog.axis_data.beam_hold.actual
num_holds = len(np.where(beam_holds == 1)[0])
print(f"Number of Beam Holds: {num_holds}")

# Calculate beam on/off transitions
print("\n=== Beam On/Off Analysis ===")
beam_transitions = np.diff(beam_holds)
num_beam_on = len(np.where(beam_transitions == -1)[0])  # Transition from hold (1) to beam on (0)
num_beam_off = len(np.where(beam_transitions == 1)[0])  # Transition from beam on (0) to hold (1)
print(f"Number of Beam On Events: {num_beam_on}")
print(f"Number of Beam Off Events: {num_beam_off}")

# Calculate RMS, Percentage and Absolute errors for MLC leaves
print("\n=== Analyze MLC data ===")
mlc = tlog.axis_data.mlc
num_leaves = mlc.num_leaves
print(f"no of leaves moved during treatment: {tlog.axis_data.mlc.num_moving_leaves}")
leaf_rms = []
leaf_pct_error = []
leaf_abs_error = []
for leaf in range(1, num_leaves + 1):
    leaf_data = mlc.leaf_axes[leaf]
    actual = leaf_data.actual
    expected = leaf_data.expected
    difference = actual - expected
    # RMS error
    rms = ((difference ** 2).mean()) ** 0.5
    leaf_rms.append(rms)
    # Percentage error
    mask = np.abs(actual) >= 0.0001
    pct_error = np.abs(difference[mask] / actual[mask]) * 100  # Percentage
    avg_pct_error = pct_error.mean()
    leaf_pct_error.append(avg_pct_error)
    # Absolute error
    abs_error = np.abs(difference).mean()
    leaf_abs_error.append(abs_error)  
avg_rms = sum(leaf_rms) / len(leaf_rms)
max_rms = max(leaf_rms)
print(f"Average RMS Error (cm): {avg_rms:.4f}")
print(f"Maximum RMS Error (cm): {max_rms:.4f}") 
avg_pct_error = sum(leaf_pct_error) / len(leaf_pct_error)
avg_abs_error = sum(leaf_abs_error) / len(leaf_abs_error)
max_abs_error = max(leaf_abs_error)
max_abs_error_leaf = leaf_abs_error.index(max_abs_error) + 1  # Leaf number (1-based indexing)
print(f"Average Percentage Error (%): {avg_pct_error:.4f}")
print(f"Average Absolute Error (cm): {avg_abs_error:.4f}")
print(f"Maximum Absolute Error (cm): {max_abs_error:.4f}")
print(f"Leaf with Maximum absolute Error: {max_abs_error_leaf}")

# Plot MLC RMS errors
plt.figure(figsize=(10, 6))
plt.bar(range(1, num_leaves + 1), leaf_rms)
plt.xlabel("MLC Leaf Number")
plt.ylabel("RMS Error (cm)")
plt.title("MLC Leaf RMS Errors")
plt.grid(True)
plt.savefig("mlc_rms_errors.png")
plt.close()

# Plot MLC Absolute errors
plt.figure(figsize=(10, 6))
plt.bar(range(1, num_leaves + 1), leaf_abs_error)
plt.xlabel("MLC Leaf Number")
plt.ylabel("Absolute Error (cm)")
plt.title("MLC Leaf Absolute Errors")
plt.grid(True)
plt.savefig("mlc_abs_errors.png")
plt.close()

# Carriage A and B Analysis
print("\n=== Carriage A and B Position Analysis ===")
carriages = ['carriage_A', 'carriage_B']
carriage_pct_error = {}
carriage_abs_error = {}
for carriage in carriages:
    carriage_data = getattr(tlog.axis_data, carriage)
    actual = carriage_data.actual
    expected = carriage_data.expected
    difference = actual - expected
    # Percentage error
    mask = np.abs(actual) >= 0.0001
    pct_error = np.abs(difference[mask] / actual[mask]) * 100  # Percentage
    avg_pct_error = pct_error.mean()
    carriage_pct_error[carriage] = avg_pct_error
    # Absolute error
    abs_error = np.abs(difference).mean()
    carriage_abs_error[carriage] = abs_error
    print(f"{carriage.upper()} Percentage Error (%): {avg_pct_error:.4f}")
    print(f"{carriage.upper()} Absolute Error (cm): {abs_error:.4f}")
avg_carriage_pct_error = sum(carriage_pct_error.values()) / len(carriage_pct_error)
avg_carriage_abs_error = sum(carriage_abs_error.values()) / len(carriage_abs_error)
print(f"Average Carriage Percentage Error (%): {avg_carriage_pct_error:.4f}")
print(f"Average Carriage Absolute Error (cm): {avg_carriage_abs_error:.4f}")

# Plot Carriage Position Differences
plt.figure(figsize=(10, 6))
for carriage in carriages:
    carriage_data = getattr(tlog.axis_data, carriage)
    difference = carriage_data.actual - carriage_data.expected
    plt.plot(difference, label=f"{carriage.upper()} Difference")
plt.title("Carriage Position Differences vs. Snapshot")
plt.xlabel("Snapshot Index")
plt.ylabel("Carriage Position Difference (cm)")
plt.legend()
plt.grid(True)
plt.savefig("carriage_diff.png")
plt.close()

# Plot Actual vs. Expected Carriage Positions
plt.figure(figsize=(10, 6))
for carriage in carriages:
    carriage_data = getattr(tlog.axis_data, carriage)
    plt.plot(carriage_data.actual, label=f"{carriage.upper()} Actual", linestyle="-")
    plt.plot(carriage_data.expected, label=f"{carriage.upper()} Expected", linestyle="--")
plt.title("Carriage Positions vs. Snapshot")
plt.xlabel("Snapshot Index")
plt.ylabel("Carriage Position (cm)")
plt.legend()
plt.grid(True)
plt.savefig("carriage_angle.png")
plt.close()

# Jaw Analysis
print("\n=== Jaw Position Analysis ===")
jaws = ['x1', 'x2', 'y1', 'y2']
jaw_pct_error = {}
jaw_abs_error = {}
for jaw in jaws:
    jaw_data = getattr(tlog.axis_data.jaws, jaw)
    actual = jaw_data.actual
    expected = jaw_data.expected
    difference = actual - expected
    # Percentage error
    mask = np.abs(actual) >= 0.0001
    pct_error = np.abs(difference[mask] / actual[mask]) * 100  # Percentage
    avg_pct_error = pct_error.mean()
    jaw_pct_error[jaw] = avg_pct_error
    # Absolute error
    abs_error = np.abs(difference).mean()
    jaw_abs_error[jaw] = abs_error
    print(f"{jaw.upper()} Jaw Percentage Error (%): {avg_pct_error:.4f}")
    print(f"{jaw.upper()} Jaw Absolute Error (cm): {abs_error:.4f}")
avg_jaw_pct_error = sum(jaw_pct_error.values()) / len(jaw_pct_error)
avg_jaw_abs_error = sum(jaw_abs_error.values()) / len(jaw_abs_error)
print(f"Average Jaw Percentage Error (%): {avg_jaw_pct_error:.4f}")
print(f"Average Jaw Absolute Error (cm): {avg_jaw_abs_error:.4f}")

# Plot Jaw Position Differences
plt.figure(figsize=(10, 6))
for jaw in jaws:
    jaw_data = getattr(tlog.axis_data.jaws, jaw)
    difference = jaw_data.actual - jaw_data.expected
    plt.plot(difference, label=f"{jaw.upper()} Jaw Difference")
plt.title("Jaw Position Differences vs. Snapshot")
plt.xlabel("Snapshot Index")
plt.ylabel("Jaw Position Difference (cm)")
plt.legend()
plt.grid(True)
plt.savefig("jaw_diff.png")
plt.close()

# Plot Actual vs. Expected Jaw Positions
plt.figure(figsize=(10, 6))
for jaw in jaws:
    jaw_data = getattr(tlog.axis_data.jaws, jaw)
    plt.plot(jaw_data.actual, label=f"{jaw.upper()} Actual", linestyle="-")
    plt.plot(jaw_data.expected, label=f"{jaw.upper()} Expected", linestyle="--")
plt.title("Jaw Positions vs. Snapshot")
plt.xlabel("Snapshot Index")
plt.ylabel("Jaw Position (cm)")
plt.legend()
plt.grid(True)
plt.savefig("jaw_angle.png")
plt.close()

# Gantry Calculations Analysis
print("\n=== Gantry Angle Analysis ===")
gantry = tlog.axis_data.gantry
gantry_actual = gantry.actual
gantry_expected = gantry.expected
gantry_diff = gantry_actual - gantry_expected
# Percentage error
mask = np.abs(gantry_actual) >= 0.01
gantry_pct_error = np.abs(gantry_diff[mask] / gantry_actual[mask]) * 100  # Percentage
avg_gantry_pct_error = gantry_pct_error.mean()
print(f"Gantry Angle Mean Error (deg): {np.mean(np.abs(gantry_diff)):.3f}")
print(f"Gantry Angle Percentage Error (%): {avg_gantry_pct_error:.4f}")

# Plot Gantry angle Differences
plt.figure(figsize=(10, 6))
plt.plot(gantry_diff)
plt.title("Gantry Angle Difference vs. Snapshot")
plt.xlabel("Snapshot Index")
plt.ylabel("Gantry Angle Difference (degrees)")
plt.grid(True)
plt.savefig("gantry_diff.png")
plt.close()

# Plot gantry angle over time
plt.figure(figsize=(10, 6))
plt.plot(gantry.actual, label="Actual Gantry Angle")
plt.plot(gantry.expected, label="Expected Gantry Angle", linestyle="--")
plt.xlabel("Snapshot Index")
plt.ylabel("Gantry Angle (degrees)")
plt.title("Gantry Angle vs. Snapshot")
plt.legend()
plt.grid(True)
plt.savefig("gantry_angle.png")
plt.close()

# Collimator Angle Analysis
print("\n=== Collimator Angle Analysis ===")
collimator = tlog.axis_data.collimator
collimator_actual = collimator.actual
collimator_expected = collimator.expected
collimator_diff = collimator_actual - collimator_expected
# Percentage error
mask = np.abs(collimator_actual) >= 0.01
collimator_pct_error = np.abs(collimator_diff[mask] / collimator_actual[mask]) * 100  # Percentage
avg_collimator_pct_error = collimator_pct_error.mean()
print(f"Collimator Angle Mean Error (deg): {np.mean(np.abs(collimator_diff)):.3f}")
print(f"Collimator Angle Percentage Error (%): {avg_collimator_pct_error:.4f}")

# Plot Collimator angle Differences
plt.figure(figsize=(10, 6))
plt.plot(collimator_diff)
plt.title("Collimator Angle Difference vs. Snapshot")
plt.xlabel("Snapshot Index")
plt.ylabel("Collimator Angle Difference (degrees)")
plt.grid(True)
plt.savefig("Collimator_diff.png")
plt.close()

# Plot collimator angle over time
plt.figure(figsize=(10, 6))
plt.plot(collimator.actual, label="Actual Collimator Angle")
plt.plot(collimator.expected, label="Expected Collimator Angle", linestyle="--")
plt.xlabel("Snapshot Index")
plt.ylabel("Collimator Angle (degrees)")
plt.title("Collimator Angle vs. Snapshot")
plt.legend()
plt.grid(True)
plt.savefig("collimator_angle.png")
plt.close()

# MU Analysis
print("\n=== Monitor Unit (MU) Analysis ===")
mu = tlog.axis_data.mu
mu_actual = mu.actual
mu_expected = mu.expected
# Total MU
total_mu_a = np.max(mu_actual)
total_mu_e = np.max(mu.expected)
mu_diff = mu_actual - mu_expected
# Absolute error
mu_abs_error = np.abs(mu_diff).mean()
# Percentage error
mask = np.abs(mu_expected) >= 0.01
mu_pct_error = np.abs(mu_diff[mask] / mu_expected[mask]) * 100  # Percentage
avg_mu_pct_error = mu_pct_error.mean()
print(f"Total MU Delivered: {total_mu_a:.4f}")
print(f"Total MU Expected: {total_mu_e:.4f}")
print(f"MU Mean Absolute Error (MU): {mu_abs_error:.4f}")
print(f"MU Percentage Error (%): {avg_mu_pct_error:.4f}")

# Plot MU difference vs. snapshot
plt.figure(figsize=(10, 6))
plt.plot(mu_diff)
plt.title("Monitor Unit Difference vs. Snapshot")
plt.xlabel("Snapshot Index")
plt.ylabel("MU Difference (MU)")
plt.grid(True)
plt.savefig("mu_diff.png")
plt.close()

# Plot Actual vs. Expected MU vs. Snapshot
plt.figure(figsize=(10, 6))
plt.plot(mu_actual, label="Actual MU", linestyle="-")
plt.plot(mu_expected, label="Expected MU", linestyle="--")
plt.title("Monitor Units vs. Snapshot")
plt.xlabel("Snapshot Index")
plt.ylabel("Monitor Units (MU)")
plt.legend()
plt.grid(True)
plt.savefig("mu_actual_expected.png")
plt.close()

# Dose Rate Analysis
print("\n=== Dose Rate Analysis ===")
mu = tlog.axis_data.mu
sampling_interval = header.sampling_interval / 1000.0  # Convert msec to sec
time = np.arange(len(mu.actual)) * sampling_interval
dose_rate_actual = np.gradient(mu.actual, time) * 60.0  # MU/sec to MU/min
dose_rate_expected = np.gradient(mu.expected, time) * 60.0  # MU/sec to MU/min
dose_rate_diff = dose_rate_actual - dose_rate_expected
print(f"Dose Rate Mean Error (MU/min): {np.mean(np.abs(dose_rate_diff)):.2f}")

# Plot dose rate over time
plt.figure(figsize=(10, 6))
plt.plot(dose_rate_actual, label="Actual Dose Rate")
plt.plot(dose_rate_expected, label="Expected Dose Rate", linestyle="--")
plt.xlabel("Snapshot Index")
plt.ylabel("Dose Rate (MU/min)")
plt.title("Dose Rate vs. Snapshot")
plt.legend()
plt.grid(True)
plt.savefig("dose_rate.png")
plt.close()

# Perform gamma analysis of fluence
print("\n=== Gamma Analysis ===")
tlog.fluence.actual.calc_map()
tlog.fluence.expected.calc_map(resolution=1)
tlog.fluence.gamma.calc_map(distTA=0.1, doseTA=0.1, resolution=0.1)
print(f"Gamma Pass Rate (%): {tlog.fluence.gamma.pass_prcnt:.2f}")
print(f"Average Gamma: {tlog.fluence.gamma.avg_gamma:.4f}")

# calculate fluence of subbeams
tlog.subbeams[0].fluence.gamma.calc_map()

# Plot actual and expected and subbeam fluence maps
tlog.fluence.actual.plot_map()
tlog.fluence.expected.plot_map()
tlog.subbeams[0].fluence.actual.plot_map()