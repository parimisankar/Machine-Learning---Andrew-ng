function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% Calculating Sigmoid Function
h = sigmoid(X*theta);
% Calculating Cost Function
% for i = 1:m
%     J = J - y(i)*log(h(i)) - (1-y(i))*log(1-h(i));
% end

% Calculating Cost Function and Gradient
%A = size(theta);
%for j = 1:A(1,1)
    %J = 0;
    %for i = 1:m
    %    J = J - y(i)*log(h(i)) - (1-y(i))*log(1-h(i));
    %    grad(j) = grad(j) + (h(i)-y(i))*X(i,j);
    %end
    %grad(j) = grad(j)/m;
    %J = J/m;
%end
% Calculating Cost function and Gradient
J = ((-y)'*log(h)- (1-y)'*log(1-h))/m;
grad = (X'*(h-y))/m;
% =============================================================

end
