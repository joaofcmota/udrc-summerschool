% Compute the phase transition of nuclear norm minimization

% --------------------------------------------------------------------- 
% Parameters

% Matrix of dimensions n x n and rank k
n = 30;
k = 3;

% Number of measurements
m_vec = 1 : 10 : round(0.95*n^2);

% MONTE_CARLO
MC = 20;

% Threshold for success
eps_succ = 1e-4;
% --------------------------------------------------------------------- 

% --------------------------------------------------------------------- 
% Generate data randomly

U_gt = randn(n, k);
X_gt = U_gt*U_gt';

% --------------------------------------------------------------------- 


% --------------------------------------------------------------------- 
% CVX

n_sq  = n^2;
m_len = length(m_vec);

% To store results
errors_cvx = zeros(m_len, MC);   % Gaussian entries
errors_net = zeros(m_len, MC);   % Netflix setting

for i_m = 1 : m_len
    m = m_vec(i_m);

    fprintf('Experiment %d  (out of %d)\n', i_m, m_len)

    for i_mc = 1 : MC

        % ----------------------------------------------------------
        % Gaussian

        % Obtain measurements
        M_p = cell(m, 1);
        y_p = zeros(m, 1);
        for i_p = 1 : m
            M_p{i_p} = randn(n, n);  % Gaussian
            y_p(i_p) = trace(M_p{i_p}*X_gt);
        end

        cvx_begin quiet
            variable X_cvx(n, n);
            minimize(norm_nuc(X_cvx));
            subject to
            for ind_meas = 1 : m
                trace(M_p{ind_meas}*X_cvx) == y_p(ind_meas);
            end
        cvx_end

        error_cvx(i_m, i_mc) = norm(X_cvx - X_gt)/norm(X_gt);
        % ----------------------------------------------------------

        % ----------------------------------------------------------
        % Netflix setting

        % Obtain measurements
        M_p = cell(m, 1);
        y_p = zeros(m, 1);
        for i_p = 1 : m
            % Randomly select an observed entry
            rp = randperm(n_sq);
            ind = rp(1);
            M_p{i_p} = zeros(n, n);
            M_p{i_p}(ind) = 1;
            y_p(i_p) = trace(M_p{i_p}*X_gt);
        end

        cvx_begin quiet
            variable X_cvx(n, n);
            minimize(norm_nuc(X_cvx));
            subject to
            for ind_meas = 1 : m
                trace(M_p{ind_meas}*X_cvx) == y_p(ind_meas);
            end
        cvx_end

        error_net(i_m, i_mc) = norm(X_cvx - X_gt)/norm(X_gt);
        % ----------------------------------------------------------

    end
end
% --------------------------------------------------------------------- 

% --------------------------------------------------------------------- 
save('phase_transition_res.mat')
% --------------------------------------------------------------------- 

% --------------------------------------------------------------------- 
% Phase transition

successes    = (error_cvx <= eps_succ);
successes_av = (1/MC)*successes*ones(MC,1);

successes_net    = (error_net <= eps_succ);
successes_net_av = (1/MC)*successes_net*ones(MC,1);

th_bound = 3*k*(2*n - 1) + 1;

figure(1);clf;
plot(th_bound*[1, 1], [0, 1], 'LineWidth', 4, 'Color', [0.94, 0.85, 0.85])
hold on;
plot(m_vec, successes_av, 'ob-', 'MarkerSize', 3)
plot(m_vec, successes_net_av, 'or-', 'MarkerSize', 3)
ylim([0,1.01])
xlim([0,900])
grid on
box off
% --------------------------------------------------------------------- 


