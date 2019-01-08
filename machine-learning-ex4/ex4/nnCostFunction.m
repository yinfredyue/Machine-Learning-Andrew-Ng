function [J grad] = nnCostFunction(nn_params, ... % includes the unrolled parameters for the neural network
                                   input_layer_size, ... % the size of the input layer
                                   hidden_layer_size, ... % the size of the hidden layer, we only have one hidden layer
                                   num_labels, ... % the size of the output layer
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % the number of training examples
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% For part 1 and 2, ignore regularization

% Part 1: Forward Propagation
% The matrix multiplication down below is from ex3, predict.m
a1 = [ones(m,1) X]';
z2 = Theta1 * a1;
a2 = [ ones(1,size(z2,2)); sigmoid(z2)]; 
% Note that sigmoid activation function should not be applied to bias unit.
z3 = Theta2 * a2;
a3 = sigmoid(z3); % 10 x 5000
h_theta_x = a3'; % 5000 x 10, Each row contains a prediction.

% IMPORTANT: recode y to get Y
% y is a 5000-dimensional vector.  
% In the step of computing J, Y should have the same size as h_theta_x: num_labels x m.
% The following is a clever way of transfering y into Y. 
% Without this step, no error will be thrown but the result is wrong. 
Y = zeros(size(h_theta_x));
I = eye(num_labels);
for i=1:m, % iterating throught y
   Y(i, :) = I( y(i) , :); % labels in rows
end  

J = (-1/m) * sum( sum( Y .* log(h_theta_x) + (1-Y) .* log(1-h_theta_x),2 ) );

% Part 2: Back Propagation
for t=1:m,
  % Forward propagation, similar to part 1
  X_with_bias_unit = [ones(m,1) X];
  a_1 = X_with_bias_unit (t,:)'; % a_1 is a column vector
  z_2 = Theta1*a_1; % z_2: 25 x 1
  a_2 = [1; sigmoid(z_2)]; % a_2: 26 x 1
  z_3 = Theta2*a_2; % z_3 = 10 x 1
  a_3 = sigmoid(z_3); % a_3: 10 x 1
  
  y_i = Y(t,:)'; % y_i = 10 x 1
  
  % Compute delta of output layer
  delta_3 = a_3 - y_i; % delta_3 is a column vector, 10 x 1
  
  % Compute delta for hidden layer(s)
  delta_2 = (Theta2' * delta_3)(2:end) .* sigmoidGradient(z_2);
  
  % delta_3: 10 x 1;
  % delta_2: 25 x 1;
  % Theta1, Theta1_grad: 25 x 401;
  % Theta2, Theta2_grad: 10 x 26;
  
  % Accumulate gradient
  Theta2_grad = Theta2_grad + delta_3 * a_2';
  Theta1_grad = Theta1_grad + delta_2 * a_1';
end

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

% regularization
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end); 
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);

% Part 3: Regularization
penalty = sum(sum( Theta1(:,2:end) .^ 2 )) + sum(sum( Theta2(:,2:end) .^ 2 ));
J = J + lambda/2/m*penalty;

% Why the following code does not work?
%Theta = {Theta1; Theta2};
%penalty = 0.0;
%for i = 1:length(Theta),
  %penalty = penalty + (sum(sum(Theta{i}(:,2:end)) .^ 2)); 
%end
%J = J + lambda/2/m*penalty;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
