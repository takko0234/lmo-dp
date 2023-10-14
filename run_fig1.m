% Define the range for q and k
clear all
clc

q_values = logspace(-2, -4, 2);
k_values = floor(logspace(1, 2, 5));

% Number of samples to generate for each pair
num_samples = 2000;

% Initialize arrays to store KL divergence, L2 metric, and EMD values
kl_divergence = zeros(length(q_values), length(k_values));
l2_metric = zeros(length(q_values), length(k_values));
emd_metric = zeros(length(q_values), length(k_values));

% Analyze Simulated Space (Example: Calculate Mean and Variance)
simulated_mean = 0.5;  % Adjust based on the characteristics of your simulated space
simulated_variance = 0.1;  % Adjust based on the characteristics of your simulated space

% Generate adaptive LMO samples for each q and k
for q_idx = 1:length(q_values)
    q = q_values(q_idx);
    
    for k_idx = 1:length(k_values)
        k = k_values(k_idx);
        
        % Simulated space: Generate PMFs
        kl_samples = zeros(num_samples, k);
        N = 1 / q;
        p = 1 / k;
        for sample_idx = 1:num_samples
            sample = mnrnd(N, ones(1, k) * p) / N;  % Multinomial sampling and normalization
            kl_samples(sample_idx, :) = sample;
        end
        
        % Estimate LMO Parameters (Example: Laplace Distribution)
        lmo_mean = simulated_mean;  % Match mean
        lmo_variance = simulated_variance;  % Match variance
        
        % Avoid extremely small variances that may lead to complex numbers
        min_variance = 1e-5;
        if lmo_variance < min_variance
            lmo_variance = min_variance;
        end
        
        lmo_scale = sqrt(lmo_variance / 2);  % Adjust scale based on variance
        
        % Generate Adaptive LMO Samples
        num_adaptive_samples = 2000;  % Adjust as needed
        adaptive_lmo_samples = laprnd(num_adaptive_samples, k, lmo_mean, lmo_scale);
        
        % Calculate KL Divergence between adaptive LMO and simulated space
        kl_divergence(q_idx, k_idx) = calculate_kl_divergence(kl_samples, adaptive_lmo_samples, num_samples);
        
        % Calculate L2 Metric (Euclidean Distance)
        l2_metric(q_idx, k_idx) = calculate_l2_metric(kl_samples, adaptive_lmo_samples, num_samples);
        
        % Calculate Earth Mover's Distance (EMD)
        emd_metric(q_idx, k_idx) = calculate_emd_metric(kl_samples, adaptive_lmo_samples, num_samples);
    end
end

% Normalize KL divergence, L2 metric, and EMD values
kl_divergence = kl_divergence ./ num_samples;
l2_metric = l2_metric ./ num_samples;
emd_metric = emd_metric ./ num_samples;

% Create three separate plots for KL divergence, L2 metric, and EMD
figure;

% Plot KL divergence
subplot(1, 3, 1);
for q_idx = 1:length(q_values)
    plot(k_values, kl_divergence(q_idx, :), '-o', 'DisplayName', ['q=' num2str(q_values(q_idx))]);
    hold on;
end
xlabel('Domain Size (k)');
ylabel('KL Divergence');
title('KL Divergence Between Adaptive LMO and Simulated Spaces');
legend('Location', 'Best');
grid on;

% Plot L2 metric (Euclidean distance)
subplot(1, 3, 2);
for q_idx = 1:length(q_values)
    plot(k_values, l2_metric(q_idx, :), '-o', 'DisplayName', ['q=' num2str(q_values(q_idx))]);
    hold on;
end
xlabel('Domain Size (k)');
ylabel('L2 Metric (Euclidean Distance)');
title('L2 Metric Between Adaptive LMO and Simulated Spaces');
legend('Location', 'Best');
grid on;

% Plot Earth Mover's Distance (EMD)
subplot(1, 3, 3);
for q_idx = 1:length(q_values)
    plot(k_values, emd_metric(q_idx, :), '-o', 'DisplayName', ['q=' num2str(q_values(q_idx))]);
    hold on;
end
xlabel('Domain Size (k)');
ylabel('Earth Mover''s Distance (EMD)');
title('EMD Between Adaptive LMO and Simulated Spaces');
legend('Location', 'Best');
grid on;

% Define a function to calculate KL Divergence
function kl_divergence = calculate_kl_divergence(simulated_samples, lmo_samples, num_samples)
    kl_divergence = 0;
    for sample_idx = 1:num_samples
        simulated_pmf = simulated_samples(sample_idx, :);
        lmo_sample = lmo_samples(sample_idx, :);
        
        % Calculate KL divergence between simulated_pmf and lmo_sample
        kl_divergence = kl_divergence + sum(simulated_pmf .* log(simulated_pmf ./ lmo_sample), 'omitnan');
    end
    kl_divergence = kl_divergence / num_samples;
end

% Define a function to calculate L2 Metric (Euclidean Distance)
function l2_metric = calculate_l2_metric(simulated_samples, lmo_samples, num_samples)
    l2_metric = 0;
    for sample_idx = 1:num_samples
        simulated_pmf = simulated_samples(sample_idx, :);
        lmo_sample = lmo_samples(sample_idx, :);
        
        % Calculate L2 Metric (Euclidean Distance)
        l2_metric = l2_metric + norm(simulated_pmf - lmo_sample);
    end
    l2_metric = l2_metric / num_samples;
end

% Define a function to calculate Earth Mover's Distance (EMD)
function emd_metric = calculate_emd_metric(simulated_samples, lmo_samples, num_samples)
    emd_metric = 0;
    for sample_idx = 1:num_samples
        simulated_pmf = simulated_samples(sample_idx, :);
        lmo_sample = lmo_samples(sample_idx, :);
        
        % Calculate Earth Mover's Distance (EMD)
        emd_metric = emd_metric + computeEMD(simulated_pmf, lmo_sample);
    end
    emd_metric = emd_metric / num_samples;
end

function emd_distance = computeEMD(pdf1, pdf2)
    % Ensure that the input PDFs have the same length
    if length(pdf1) ~= length(pdf2)
        error('Input PDFs must have the same length.');
    end

    % Normalize the PDFs so that they sum to 1 (probability mass function)
    pdf1 = pdf1 / sum(pdf1);
    pdf2 = pdf2 / sum(pdf2);

    % Compute the cumulative distribution functions (CDFs) of the PDFs
    cdf1 = cumsum(pdf1);
    cdf2 = cumsum(pdf2);

    % Calculate the EMD by computing the L1 distance between CDFs
    emd_distance = sum(abs(cdf1 - cdf2));
end

