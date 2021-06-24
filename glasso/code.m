% ----------------------------------------------------
% Load data

filename = 'data/data_processed.csv';
delimiterIn = ',';
headerlinesIn = 0;
data = importdata(filename,delimiterIn,headerlinesIn);
% ----------------------------------------------------

% ----------------------------------------------------
% Parameters

% Number of companies in dataset
d_orig = 30;

% Select only the first d companies (out of 30)
d =  6;  % Number of companies
n = 25;  % Number of samples    
% (toy example as n is too small to apply CLT)

% Range of value of regularization parameter
lambda_vec = 0:30;
lambda_len = length(lambda_vec);

% Threshold in precision matrix for determining an edge
eps_edge = 1e-3;
% ----------------------------------------------------

% ----------------------------------------------------
% Processing

% Reorganize data into an n x d matrix
X_orig = reshape(data.data, n, d_orig);
X_d    = X_orig(:,1:d);

% Remove the mean of each column
X = X_d - ones(n,1)*mean(X_d);

% Sample covariance
S = (1/n)*X'*X;

% Check that it is invertible
if rank(S) < d
    error('Singular sample covariance')
end

% Maximum likelihood estimator
T_MLE = inv(S);
% ----------------------------------------------------

% ----------------------------------------------------
% Graphical Lasso

% Store results
T_results = cell(lambda_len, 1);
num_edges = zeros(lambda_len, 1);

for iter = 1 : lambda_len
    lambda = lambda_vec(iter);
    cvx_begin quiet
        variable T(d,d) semidefinite
        minimize(trace(T*S)-log_det(T) + lambda*norm(T - diag(diag(T)),1))
    cvx_end

    % Store results
    T_results{iter} = T;
    num_edges(iter) = sum(sum(abs(triu(T,1)) > eps_edge));
end
% ----------------------------------------------------

% ----------------------------------------------------
% Visualization

% num_edges
figure(1);clf;
plot(lambda_vec, num_edges, 'o-')
box off
grid on

l_example = 22;
fprintf('Example of precision matrix for lambda = %d\n', lambda_vec(l_example))
T_example = T_results{l_example}

% Correlations
total_edges  = (d-1)*d/2;
correlations = zeros(total_edges, lambda_len);

for l_counter = 1 : lambda_len
    % Find nonzero indices of edge matrix
    ind = find(triu(T_results{l_counter}, 1));
    correlations(:, l_counter) = max(abs(T_results{l_counter}(ind)), eps_edge);
end

map = colormap(cool(total_edges));
figure(2);clf
semilogy(lambda_vec, eps_edge*ones(1, lambda_len), '-k', 'LineWidth', 2.2)
hold on;
plot([l_example, l_example], [eps_edge, 99], 'LineWidth', 4, 'Color', [0.94, 0.85, 0.85])
for edge_counter = 1 : total_edges
    semilogy(lambda_vec, correlations(edge_counter, :), '-o', 'Color', map(edge_counter,:))
end
ylim([1e-3,1e2])
box off
grid on

% ----------------------------------------------------

