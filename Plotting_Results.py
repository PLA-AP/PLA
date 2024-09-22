import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Function to load JSON data from the results directory
def load_json_data(results_path):
    json_data = {}
    for trial_dir in os.listdir(results_path):
        trial_path = os.path.join(results_path, trial_dir)
        if os.path.isdir(trial_path):
            print(f"Loading trial: {trial_dir}")
            trial_data = {}
            for model_dir in os.listdir(trial_path):
                model_path = os.path.join(trial_path, model_dir)
                if os.path.isdir(model_path):
                    print(f"  Loading model: {model_dir}")
                    model_data = {}
                    for result_file in os.listdir(model_path):
                        result_path = os.path.join(model_path, result_file)
                        if result_file.endswith('.json'):
                            method = result_file.split('_')[0]
                            with open(result_path, 'r') as f:
                                print(f"    Loaded file: {result_file} for method: {method}")
                                model_data[method] = json.load(f)
                    print(f"  Final model data keys for {model_dir}: {model_data.keys()}")
                    trial_data[model_dir] = model_data
            print(f"  Final trial data keys for {trial_dir}: {trial_data.keys()}")
            json_data[trial_dir] = trial_data
    return json_data

# Function to plot Figure 7
def plot_fig7(json_data):
    print("Starting to plot Figure 7...")

    # Define the feature selection methods and classifiers for the plot
    methods = ['anova', 'mutual_info', 'pca', 'rfe']
    classifiers = ['random_forest', 'svc', 'xgb', 'logistic_regression', 'knn']
    classifier_labels = ['RnF', 'SVM', 'XGB', 'LR', 'KNN']

    # Initialize a dictionary to store the ADR data per trial
    data_per_trial = {'trial_1': [], 'trial_2': [], 'trial_3': []}

    # Function to extract ADR data from JSON
    def extract_adr_data(json_data, trial, classifier, method):
        # Adjust method name if mutual_info is stored as mutual
        if method == 'mutual_info':
            method = 'mutual'
        
        if trial in json_data:
            trial_data = json_data[trial]
            if classifier in trial_data:
                classifier_data = trial_data[classifier]
                if method in classifier_data:
                    return classifier_data[method].get('overall_adr', 0) * 100  # Convert to percentage
                else:
                    print(f"  Warning: No method {method} for classifier {classifier} in {trial}")
            else:
                print(f"  Warning: No classifier {classifier} in {trial}")
        else:
            print(f"  Warning: No trial {trial} in data")
        
        return 0  # Return 0 if data is missing

    # Iterate over trials and classifiers to extract ADR from JSON data
    for trial in data_per_trial.keys():
        for classifier in classifiers:
            adr_values = []
            for method in methods:
                adr = extract_adr_data(json_data, trial, classifier, method)
                adr_values.append(adr)
            data_per_trial[trial].append(adr_values)

    # Plot the figure with three subplots for the three trials
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Set positions for bars
    bar_width = 0.2
    index = np.arange(len(classifiers))

    # Iterate over trials and plot data in each subplot
    for i, trial in enumerate(data_per_trial.keys()):
        data = np.array(data_per_trial[trial])

        for j in range(data.shape[1]):
            axs[i].bar(index + j * bar_width, data[:, j], bar_width, label=methods[j].upper())

        # Set x-axis labels to classifiers
        axs[i].set_xticks(index + bar_width * (data.shape[1] - 1) / 2)
        axs[i].set_xticklabels(classifier_labels)

        # Add labels and title
        axs[i].set_xlabel('ML Models')
        axs[i].set_ylabel('Average Detection Rate (%)')
        axs[i].set_ylim([50, 100])
        axs[i].legend(title='Feature Selection Methods', loc='upper right')
        axs[i].grid(True)

        # Add subplot title
        axs[i].set_title(f'Trial {i + 1}')

    plt.tight_layout()
    print("Figure 7 plotted successfully.")

# Function to plot Figure 8
def plot_fig8(json_data):
    print("Starting to plot Figure 8...")

    # Define the device names for the three sets (scenarios)
    x_labels1 = ['device 3', 'device 2', 'device 12', 'device 9']
    x_labels2 = ['device 10', 'device 6', 'device 12', 'device 3', 'device 11', 'device 1']
    x_labels3 = ['device 1', 'device 10', 'device 9', 'device 8', 'device 12', 'device 11', 'device 6', 'device 7']

    # Define x-coordinates for the three sets
    x1 = np.arange(1, len(x_labels1) + 1) - 0.1
    x2 = np.arange(1, len(x_labels2) + 1) + max(x1) + 1
    x3 = np.arange(1, len(x_labels3) + 1) + max(x2) + 0.8

    # Define the models and methods for the four subplots
    models_methods = [
        ('logistic_regression', 'anova'),  # (a) LR-ANOVA
        ('logistic_regression', 'pca'),    # (b) LR-PCA
        ('random_forest', 'mutual_info'),  # (c) RnF-MI
        ('random_forest', 'anova')         # (d) RnF-ANOVA
    ]

    # Function to extract data for devices from JSON within a specific trial
    def extract_data_for_trial(trial_devices, device_list):
        tdr_values = []
        fdr_values = []
        rogue_tdr_values = []
        
        print(f"Extracting data for devices: {device_list}")
        print(f"Available device keys in trial data: {trial_devices.keys()}")

        for device in device_list:
            device_key = device.replace(" ", "").lower()  # Normalize device name
            print(f"  Looking for device: {device_key}")
            if device_key in trial_devices:
                device_data = trial_devices[device_key]
                tdr_values.append(device_data['auth_tvr'] * 100)
                fdr_values.append(device_data['auth_fvr'] * 100)
                rogue_tdr_values.append(device_data['rogue_tvr'] * 100)
                print(f"    Found data for {device_key}: {device_data}")
            else:
                print(f"  Warning: Data for {device} not found in this trial.")
                tdr_values.append(0)
                fdr_values.append(0)
                rogue_tdr_values.append(0)
        
        return tdr_values, fdr_values, rogue_tdr_values

    # Function to extract data from a specific trial
    def extract_trial_data(json_data, model, method, x_labels, trial_key):
        print(f"Extracting data for trial: {trial_key}, model: {model}, method: {method}")
        
        # Adjust method name if mutual_info is stored as mutual
        if method == 'mutual_info':
            method = 'mutual'
        
        if trial_key in json_data:
            trial_data = json_data[trial_key]
            print(f"  Trial {trial_key} contains models: {trial_data.keys()}")
            if model in trial_data:
                model_data = trial_data[model]
                print(f"    Model {model} contains methods: {model_data.keys()}")
                if method in model_data:
                    method_data = model_data[method]
                    trial_devices = method_data.get('devices', {})
                    return extract_data_for_trial(trial_devices, x_labels)
                else:
                    print(f"  Warning: No method {method} in model {model}")
            else:
                print(f"  Warning: No model {model} in trial {trial_key}")
        else:
            print(f"  Warning: No trial {trial_key} found in data")
        
        return [0] * len(x_labels), [0] * len(x_labels), [0] * len(x_labels)

    # Create figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot data for each model-method combination
    for idx, (model, method) in enumerate(models_methods):
        ax = axs[idx // 2, idx % 2]
        
        # Extract data for each trial
        True1, Others_A1, True_rogue1 = extract_trial_data(json_data, model, method, x_labels1, 'trial_1')
        True2, Others_A2, True_rogue2 = extract_trial_data(json_data, model, method, x_labels2, 'trial_2')
        True3, Others_A3, True_rogue3 = extract_trial_data(json_data, model, method, x_labels3, 'trial_3')

        # Ensure the length of x-coordinates matches the number of data points
        if len(True1) != len(x1) or len(True2) != len(x2) or len(True3) != len(x3):
            print(f"Error: Data length mismatch in scenario {idx+1}")
            continue

        # Print debug information
        print(f"\nModel: {model}, Method: {method}")
        print(f"True1: {True1}, Others_A1: {Others_A1}, True_rogue1: {True_rogue1}")
        print(f"True2: {True2}, Others_A2: {Others_A2}, True_rogue2: {True_rogue2}")
        print(f"True3: {True3}, Others_A3: {Others_A3}, True_rogue3: {True_rogue3}")

        # Function to plot the data
        def plot_data(x, true_vals, others_a, true_rogue):
            ax.plot(x, true_vals, 'o', markersize=10, markeredgecolor='b', markerfacecolor='b')
            ax.plot(x, others_a, 's', markersize=14, markeredgecolor='black', markerfacecolor='#EDB120')
            ax.plot(x, true_rogue, 'x', markersize=10, markeredgecolor='r', linewidth=2)

        # Plot data for each set of devices
        plot_data(x1, True1, Others_A1, True_rogue1)
        plot_data(x2, True2, Others_A2, True_rogue2)
        plot_data(x3, True3, Others_A3, True_rogue3)

        # Add vertical dashed lines to separate the scenarios
        ax.axvline(x=max(x1) + 1, color='k', linestyle='--')
        ax.axvline(x=max(x2) + 1, color='k', linestyle='--')

        # Set xticks and labels
        ax.set_xticks(np.concatenate([x1, x2, x3]))
        ax.set_xticklabels(x_labels1 + x_labels2 + x_labels3, rotation=45)

        # Set y-limits and grid lines
        ax.set_ylim([0, 100])
        ax.axhline(y=5, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=95, color='gray', linestyle='--', linewidth=1)

        # Add legend in the first plot
        if idx == 0:
            ax.legend(['Authorized Detected', 'Authorized Missed', 'Malicious Detected'], loc='upper right')

        # Add scenario labels
        ax.text(np.mean(x1), 50, 'Scenario 1', horizontalalignment='center')
        ax.text(np.mean(x2), 50, 'Scenario 2', horizontalalignment='center')
        ax.text(np.mean(x3), 50, 'Scenario 3', horizontalalignment='center')

        # Set title for each subplot
        ax.set_title(f'{model.upper()}-{method.upper()}')

    plt.tight_layout()
    print("Figure 8 plotted successfully.")

# Function to calculate miss detection rates
def calculate_mdr(devices):
    # Sum of FDR for authorized devices
    FDR_auth_sum = sum(device['auth_fvr'] for device in devices.values())
    # Sum of (1 - TDR) for rogue devices
    missed_rogue_sum = sum(1 - device['rogue_tvr'] for device in devices.values())
    num_devices = len(devices)

    # Calculate miss detection rates
    missed_authorized = FDR_auth_sum / num_devices
    missed_malicious = missed_rogue_sum / num_devices
    
    return missed_authorized, missed_malicious

# Function to plot Figure 9
def plot_fig9(json_data):
    print("Starting to plot Figure 9...")

    classifiers = ['random_forest', 'logistic_regression']
    methods = {'random_forest': ['anova', 'mutual'], 'logistic_regression': ['anova', 'pca']}  # Only four combinations
    classifier_labels = ['RnF-ANOVA', 'RnF-MI', 'LR-ANOVA', 'LR-PCA']  # Just the 4 combinations

    # Initialize dictionaries for storing missed detection rates
    missedAuthorized_rates = {'trial_1': [], 'trial_2': [], 'trial_3': []}
    missedMalicious_rates = {'trial_1': [], 'trial_2': [], 'trial_3': []}

    # Debug: Print available trials
    print(f"Available trials: {json_data.keys()}")

    # Iterate through the trials, classifiers, and methods to extract missed detection rates
    for trial in missedAuthorized_rates.keys():
        print(f"\nProcessing {trial}...")
        for classifier in classifiers:
            for method in methods[classifier]:  # Limit methods based on the dictionary
                print(f"Processing {classifier} with {method}...")
                if classifier in json_data[trial]:
                    if method in json_data[trial][classifier]:
                        devices = json_data[trial][classifier][method]['devices']
                        missedAuthorized, missedMalicious = calculate_mdr(devices)
                        missedAuthorized_rates[trial].append(missedAuthorized * 100)  # Convert to percentage
                        missedMalicious_rates[trial].append(missedMalicious * 100)
                        print(f"  Found data for {classifier}-{method} in {trial}: Missed Authorized = {missedAuthorized*100}, Missed Malicious = {missedMalicious*100}")
                    else:
                        print(f"  No method {method} for {classifier} in {trial}, appending 0.")
                        missedAuthorized_rates[trial].append(0)
                        missedMalicious_rates[trial].append(0)
                else:
                    print(f"  No classifier {classifier} in {trial}, appending 0.")
                    missedAuthorized_rates[trial].append(0)
                    missedMalicious_rates[trial].append(0)

        # Debug: Print collected missed detection rates for the trial
        print(f"  Missed Authorized Rates for {trial}: {missedAuthorized_rates[trial]}")
        print(f"  Missed Malicious Rates for {trial}: {missedMalicious_rates[trial]}")

    # Plotting the figure with 3 subplots for the 3 trials
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    bar_width = 0.35
    index = np.arange(len(classifier_labels))  # Should now be 4

    for i, trial in enumerate(missedAuthorized_rates.keys()):
        ax = axs[i]
        
        # Debug: Print lengths of index and missed detection rates before plotting
        print(f"Plotting {trial}... index length: {len(index)}, missedAuthorized length: {len(missedAuthorized_rates[trial])}, missedMalicious length: {len(missedMalicious_rates[trial])}")

        # Plot missed authorized and missed malicious bars side by side
        ax.bar(index, missedAuthorized_rates[trial], bar_width, color=[0, 0.4470, 0.7410], label='Missed Authorized')
        ax.bar(index + bar_width, missedMalicious_rates[trial], bar_width, color=[0.8500, 0.3250, 0.0980], label='Missed Malicious')

        # Set x-axis labels
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(classifier_labels, rotation=45)
        
        # Set labels and limits
        ax.set_ylabel('Missed Detection Rate (%)')
        ax.set_ylim([0, 20])

        # Set y-ticks with increments of 5 and ensure integer formatting
        ax.set_yticks(np.arange(0, 21, 5))  # Set y-ticks from 0 to 20 with a step of 5
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))  # Format y-ticks as integers

        # Add title for each trial
        ax.set_title(f'Trial {i + 1}')

    # Add legend
    axs[0].legend(loc='best')

    plt.tight_layout()
    print("Figure 9 plotted successfully.")

# Function to load SNR data from directory structure
def load_snr_data(results_path):
    snr_data = {}
    
    for snr_dir in os.listdir(results_path):
        snr_path = os.path.join(results_path, snr_dir)
        if os.path.isdir(snr_path):
            snr_file = os.path.join(snr_path, f"{snr_dir}.json")
            if os.path.exists(snr_file):
                with open(snr_file, 'r') as f:
                    data = json.load(f)
                    snr_value = data.get('snr', None)
                    avg_detection_rates = data.get('avg_detection_rates', None)
                    if snr_value is not None and avg_detection_rates is not None:
                        snr_data[snr_value] = avg_detection_rates
                    else:
                        print(f"Warning: SNR or avg_detection_rates missing in {snr_file}")
            else:
                print(f"Warning: {snr_file} not found")
    
    return snr_data

# Function to plot Figure 10 (SNR vs Average Detection Rate for different scenarios)
def plot_fig10(results_path):
    print("Starting to plot Figure 10...")
    snr_data = load_snr_data(results_path)
    
    # Extract SNR values and average detection rates
    snr_values = sorted(snr_data.keys())
    
    scenario1_rates = [snr_data[snr][0] for snr in snr_values]  # First scenario
    scenario2_rates = [snr_data[snr][1] for snr in snr_values]  # Second scenario
    scenario3_rates = [snr_data[snr][2] for snr in snr_values]  # Third scenario
    
    plt.figure(figsize=(6, 4))
    
    # Plotting the scenarios
    plt.plot(snr_values, scenario1_rates, 'r-', linewidth=2, label='Scenario 1')
    plt.plot(snr_values, scenario2_rates, 'g-', linewidth=2, label='Scenario 2')
    plt.plot(snr_values, scenario3_rates, 'b-', linewidth=2, label='Scenario 3')
    
    # Labels and legend
    plt.xlabel('SNR (dB)')
    plt.ylabel('Average Detection Rate (%)')
    plt.legend()
    
    # Set axis limits
    plt.xlim([min(snr_values), max(snr_values)])
    plt.ylim([min(min(scenario1_rates), min(scenario2_rates), min(scenario3_rates)) - 1, 100])
    
    # Add grid
    plt.grid(True)
    
    plt.tight_layout()
    print("Figure 10 plotted successfully.")
# Main function to execute the workflow
def main():
    results_path = "results"
    json_data = load_json_data(results_path)
    results_path_noise = 'results_noise'

    # Plotting figure 7
    plot_fig7(json_data)

    # Plotting figure 8
    plot_fig8(json_data)

    # Plotting figure 9
    plot_fig9(json_data)

    # Plotting figure 10
    plot_fig10(results_path_noise)

    # Display all figures
    plt.show()

# Entry point for the script
if __name__ == "__main__":
    main()
