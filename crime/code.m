% ----------------------------------------------------
% Load data (preprocessed to csv)

filename = 'data/crime.csv';
delimiterIn = ',';
headerlinesIn = 0;
data = importdata(filename,delimiterIn,headerlinesIn);
% ----------------------------------------------------

% --------------------------------
% Form matrix A and measurements y
A_init = data(:,3:end);
y_init = data(:,1);

[n, d] = size(A_init);

% Add a constant column of ones to A
A = [ones(n, 1), A_init];
y = y_init;
d = d + 1;
% --------------------------------

% --------------------------------
% Least-squares solution

x_ls = A'*A \ A'*y;
% --------------------------------

% --------------------------------
% Lasso

% Trace path of solutions from lambda = 0 to lambda = 1e4
lambda_vec = [0, 0.01, 0.1, 1, 10, 1e2, 1e3, 1e4];
lambda_len = length(lambda_vec);

results_lasso = zeros(d, lambda_len);

% To ignore the abs value of first entry of x
C = eye(d);
C(1,1) = 0;

for iter = 1 : lambda_len

    lambda = lambda_vec(iter);
    cvx_begin quiet
        variable x(d, 1);
        minimize ((1/(2*n))*square_pos(norm(A*x - y, 2)) + lambda*norm(C*x, 1));
    cvx_end

    results_lasso(:, iter) = x;
end
% --------------------------------

% --------------------------------
% Visualize

map = colormap(cool(d));
figure(1);clf;
semilogx(lambda_vec, zeros(1, lambda_len), '-k', 'LineWidth', 2.2)
hold on;
for i_d = 2 : d
    semilogx(lambda_vec, results_lasso(i_d,:), '-o', 'Color', map(i_d,:))
end
ylim([-10,15])
%xlim([0, 1e4])
grid on
box off
%legend('zero', 'funding', '4 years hs (25+)', 'no-hs (16-19)', 'college (18-24)', 'college (25+)' )
% --------------------------------
