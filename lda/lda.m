% Linear discriminant analysis in high-dimensions

% ---------------------------------------------------------------------- 
% Parameters

% Dimensions, and number of samples
d  = 400;
n1 = 800;
n2 = 800;

% Monte-Carlo experiments
MC = 5000;
% ---------------------------------------------------------------------- 

% ---------------------------------------------------------------------- 
% Theoretical expressions

threshold  = sqrt(2*log(d)/n1);
alpha = d/n1;

mean_shift = 1:0.01:2;
length_exp = length(mean_shift);

% Create standard normal distribution
mu_stn    = 0;
sigma_stn = 1;
pd = makedist('Normal', mu_stn, sigma_stn);

% Classical expression
prob_error_classical = cdf(pd, -mean_shift/2);

% Kolmogorov expression
prob_error_high_dimensional = zeros(length_exp, 1);
x_high_dimensional = -(mean_shift.^2)./(2*sqrt(mean_shift.^2 + 2*alpha));
prob_error_high_dimensional = cdf(pd, x_high_dimensional);
% ---------------------------------------------------------------------- 

% ---------------------------------------------------------------------- 
% Experimental

mean_shift_exp = 1:0.1:2;
length_exp_exp = length(mean_shift_exp);

% mu1 is zero; and mu2 = mean_shift(i)*random_direction
mu1 = zeros(d,1);

test_result = zeros(length_exp_exp, MC);
true_distri = zeros(length_exp_exp, MC);

for i = 1 : length_exp_exp
    % Compute mu2
    mean_shift_i = mean_shift_exp(i);
    random_direction_aux = randn(d,1);
    random_direction     = random_direction_aux/norm(random_direction_aux);
    mu2 = mean_shift_i*random_direction;

    % Generate samples
    X1 = randn(d, n1);
    X2 = mu2*ones(1,n2) + randn(d, n2);

    % Estimate sample means
    mu1_est = X1*ones(n1,1)./n1;
    mu2_est = X2*ones(n2,1)./n2;

    for mc = 1 : MC
        % Select P1 or P2 with equal probability
        if rand < 0.5
            % P1
            true_distri(i, mc) = 1;
            x_sample = randn(d, 1);

            % Fisher linear discriminant function
            if (x_sample - 0.5*(mu1_est + mu2_est))'*(mu2_est - mu1_est) >= 0
                % Decide on P2, thus error
                test_result(i, mc) = 1;
            else
                test_result(i, mc) = 0;
            end

        else
            % P2
            true_distri(i, mc) = 2;
            x_sample = mu2 + randn(d, 1);

            % Fisher linear discriminant function
            if (x_sample - 0.5*(mu1_est + mu2_est))'*(mu2_est - mu1_est) >= 0
                % Decide on P2, thus good
                test_result(i, mc) = 0;
            else
                test_result(i, mc) = 1;
            end

        end
    end
end

% Compute average error
prob_error_exp = test_result*ones(MC,1)/MC;

% ---------------------------------------------------------------------- 

figure(1);clf;
plot(mean_shift, prob_error_classical, '-b', 'LineWidth', 1.5)
hold on;
plot(mean_shift_exp, prob_error_exp(1:length_exp_exp), 'ok','LineWidth', 1.1)
plot(mean_shift, prob_error_high_dimensional, '-r', 'LineWidth', 1.5)
ylim([0.1,0.4])
xlim([0.98,2.02])
legend('classical', 'high-dimensional', 'experiments')



