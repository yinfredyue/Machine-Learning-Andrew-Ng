function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% We have 8 options for both C and sigma. So we compute the cost of all 64 combinations,
% and return the one that gives the smallest error. 

C_options = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_options = [0.01 0.03 0.1 0.3 1 3 10 30];
error_cv_table = zeros(3, size(C_options,1) * size(sigma_options,1)); % 3 * 64
predictions = zeros(size(yval));

i = 0;
for _C = C_options
  for _sigma = sigma_options
    i += 1;
    model = svmTrain(X , y , _C , @(xi,xj) gaussianKernel(xi,xj,_sigma));
    predictions = svmPredict(model, Xval);
    error_cv = mean(double(predictions ~= yval));
    error_cv_table(:,i) = [_C; _sigma; error_cv];
  end
end

[val index] = min(error_cv_table(3,:));

C = error_cv_table(1,index);
sigma = error_cv_table(2,index);


% =========================================================================

end
