% Reconstruction of an image
%
% Requires sparco and spgl1 toolboxes

% ====================================================================== 
% Parameters

path_to_im = 'data/HW.png';

% Number of observations (in percentage)
m_perc = 0.5;

% ====================================================================== 

% ====================================================================== 
% Preprocessing

im_orig = im2double(imread(path_to_im));
im_gray = rgb2gray(im_orig);

% % Original grayscale image
% figure(1);clf;
% imshow(im_gray)

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
semilogy(1:n,sort(abs(x_gt), 'desc'), '-', 'LineWidth', 0.1)
hold on;
semilogy(1:n, 1e-3*ones(1,n), 'r')

% -----------------------------------------------------------------------
% Take observations

% Select indices at random
m = round(m_perc*n);
rnd_p = randperm(n);
observed_indices = rnd_p(1:m);

figure(4);clf;
mask_im = zeros(M,N);
mask_im(observed_indices) = 1;
im_observed = im_crop;
im_observed(logical(mask_im)) = 0;
imshow(im_observed)

% Build operator
Phi = opRestriction(n, observed_indices);
A = opFoG(Phi, B);

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
