options = optimset('MaxIter', 50);
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% ---------------------------------------------------------------------------------------------------------------
function [J grad] = nnCostFunction(nn_params, ... % includes the unrolled parameters for the neural network
                                   input_layer_size, ... % the size of the input layer
                                   hidden_layer_size, ... % the size of the hidden layer, we only have one hidden layer
                                   num_labels, ... % the size of the output layer
                                   X, y, lambda)

% retrieve the original Thetas
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1); % the number of training examples
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Forward Propagation
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


% These are the captial Delta terms.
delta_diff_1 = zeros(size(Theta1)); 
delta_diff_2 = zeros(size(Theta2));


% Part 2: Back Propagation
for t=1:m,
  % Forward propagation to compute the output vector
  X_with_bias_unit = [ones(m,1) X];
  a_1 = X_with_bias_unit (t,:)'; % a_1 is a column vector
  z_2 = Theta1*a_1; % z_2: 25 x 1
  a_2 = [1; sigmoid(z_2)]; % a_2: 26 x 1
  z_3 = Theta2*a_2; % z_3 = 10 x 1
  a_3 = sigmoid(z_3); % a_3: 10 x 1
  
  y_i = Y(t,:)'; % y_i = 10 x 1
  
  % Compute delta of output layer
  delta_3 = a_3 - y_i; % delta_3 is a column vector, 10 x 1
  
  % Compute delta for hidden layer(s): 
  % With a neural network of L layers, we should calculate delta_j, for j = 2, 3, 4, ..., L.
  delta_2 = (Theta2' * delta_3)(2:end) .* sigmoidGradient(z_2);
  
  % delta_3: 10 x 1;
  % delta_2: 25 x 1;
  % Theta1, Theta1_grad: 25 x 401;
  % Theta2, Theta2_grad: 10 x 26;
  
  % Accumulate gradient: the unfamiliar part.
  % With a neural network of L layers, we should update Thetaj_grad, for j = 1, 2,..., L-1.
  delta_diff_2 = delta_diff_2 + delta_3 * a_2';
  delta_diff_1 = delta_diff_1 + delta_2 * a_1';
end

% Using the fact: "partial derivative of J(theta) with respect to theta^(l) == delta_diff_l."
Theta1_grad = delta_diff_1/m;
Theta2_grad = delta_diff_2/m;

% Adding the regularization term
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
