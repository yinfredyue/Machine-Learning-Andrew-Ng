function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

mu = (1/m) * (sum(X))'; % n x 1

mu_matrix = zeros(n,m);

for i=1:m
  mu_matrix(:,i) = mu;
end

% implemetaion with for loop
%for i=1:n
%  for j = 1:m
%    sigma2(i) += (X(j,i)-mu(i))^2;
%  end
%end
%sigma2 /= m;

% vectorized implemetaion
temp = (1/m) * (X'-mu_matrix)*(X'-mu_matrix)';
for i=1:n
  sigma2(i) = temp(i,i);
end

% =============================================================


end
