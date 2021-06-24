% Reconstruction of an image from Fourier measurements
%
% Requires sparco and spgl1 toolboxes

% ====================================================================== 
% Parameters

path_to_im = 'data/HW.png';

PERC = 0.4;               % Percentage of measurements around zero freq.
                          % (non-uniform sampling)

ZERO_FREQ_NEIGHB = 0.3;   % Near zero freq. means ZERO_FREQ_NEIGHB*n
                          % and it wraps around: [***        ***]
                          % (the *'s are of length ZERO_FREQ_NEIGHB*n)

% Number of observations (in percentage)
m_perc = 0.5;

% ====================================================================== 

% ====================================================================== 
% Preprocessing

im_orig = im2double(imread(path_to_im));
im_gray = rgb2gray(im_orig);

% Original grayscale image
figure(1);clf;
imshow(im_gray)

% Crop to make dimensions suitable for sparco
im_crop = im_gray(1:256, :);
figure(2);clf;
imshow(im_crop)

% Size of image
[M, N] = size(im_crop);
n = M*N;

% Vectorize image
im_gt = vec(im_crop);

% Sparco toolbox operator: wavelet transform
B = opWavelet(n, 1, 'haar', 2, 5, 'min');

% Wavelet coefficients of im_gt
x_gt = B(im_gt, 2);

% -----------------------------------------------------------------------
% Information
fprintf('\n\nImage dimension (n):     %d\n', n);
fprintf('Sparsity (thres: %.0e): %d  (%.2f perc)\n',  1e-3, ...
    sum(abs(x_gt)>1e-3), sum(abs(x_gt)>1e-3)/n*100);
% -----------------------------------------------------------------------

% Histogram of coefficients
figure(3);clf;
plot(1:n,sort(abs(x_gt), 'desc'), '-', 'LineWidth', 0.1)
box off
grid on
%hold on;
%semilogy(1:n, 1e-3*ones(1,n), 'r')

% -----------------------------------------------------------------------
% Sampling: dense around low frequencies, sparse in high frequencies

m = round(m_perc*n);

num_meas_low_freq  = round(PERC*m);
num_meas_high_freq = round((1-PERC)*m);

% Low frequencies
max_coeff_low_freq          = round(0.5*ZERO_FREQ_NEIGHB*n);
coefficients_low_freq       = -max_coeff_low_freq : max_coeff_low_freq;
permutation_low_freq        = randperm(2*max_coeff_low_freq);
num_meas_low_freq_corrected = min(2*max_coeff_low_freq, num_meas_low_freq);
selected_indices_low_freq   = mod(coefficients_low_freq(...
    permutation_low_freq(1:num_meas_low_freq_corrected))-1, n) + 1;

% High frequencies
coefficients_high_freq       = max_coeff_low_freq +1 : (n - max_coeff_low_freq - 1);
coefficients_high_freq_len   = length(coefficients_high_freq);
permutation_high_freq        = randperm(coefficients_high_freq_len);
num_meas_high_freq_corrected = min(coefficients_high_freq_len, num_meas_high_freq);
selected_indices_high_freq   = coefficients_high_freq(...
    permutation_high_freq(1:num_meas_high_freq_corrected));
% -----------------------------------------------------------------------

% Partial DFT operator
partialDFT = opFoG(opRestriction(n, ...
    [selected_indices_low_freq, selected_indices_high_freq]'), opFFT(n));

% Operator A: composition of partial DFT with Wavelet transform
A = opFoG(partialDFT, B);

% Take measurements
b = A(x_gt, 1);
% -----------------------------------------------------------------------

% ====================================================================== 

% ====================================================================== 
% Reconstruction

opts_spgl1 = spgSetParms('verbosity',1);  % Turn off the SPGL1 log output
time_BP_aux = cputime;
x_BP = spgl1(A, b, 0, 0, [], opts_spgl1);
%x_BP = real(x_BP);
time_BP = cputime - time_BP_aux;


figure(5);clf;
im_rec_vec = B(x_BP, 1);
im_rec = reshape(im_rec_vec, M, N);
imshow(im_rec);

PSNR_BP = 20*log10(sqrt(n)/norm(double(abs(im_rec_vec))-double(im_gt),'fro'));
fprintf('Error: %f\n', norm(x_BP - x_gt)/norm(x_gt))
fprintf('PSNR:  %f\n', PSNR_BP)

% ====================================================================== 
