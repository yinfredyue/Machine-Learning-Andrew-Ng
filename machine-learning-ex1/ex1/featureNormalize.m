function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly

X_norm = X;
% X_norm should be returned as the normalized matrix
mu = zeros(1, size(X, 2)); 
% mu is the pronunciation of Greek letter for mean
% 2 means the second dimension of the matrix, i.e. how many elements in a row.
sigma = zeros(1, size(X, 2));

n = size(X,2); % n is the number of features
m = size(X,1); % m is the number of training examples

% sigma is the pronunciation of Greek letter for standard deviation

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

for i=1:n,
  mu(i) = mean(X(:,i));
  sigma(i) = std(X(:,i));
end

for i=1:m,
  for j=1:n,
    X_norm(i,j) = ( X_norm(i,j) - mu(j) )/sigma(j);
  end
end

% ============================================================

end
